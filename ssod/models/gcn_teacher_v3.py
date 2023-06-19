import torch
import torch.autograd as autograd
from mmcv.runner.fp16_utils import force_fp32
from mmdet.core import bbox2roi, multi_apply, bbox_overlaps
from mmdet.models import DETECTORS, build_detector

from ssod.utils.structure_utils import dict_split, weighted_loss
from ssod.utils import log_image_with_boxes, log_every_n
from torch.nn.modules.utils import _pair
from .multi_stream_detector import MultiSteamDetector
from .utils import Transform2D, filter_invalid

@DETECTORS.register_module()
class GCNTeacherV2(MultiSteamDetector):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super(GCNTeacherV2, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:
            self.freeze("teacher")
            # unsup_weight = 2 ,for in train time
            self.unsup_weight = self.train_cfg.unsup_weight
        self.roi_feat_size = _pair(7)



    def forward_train(self, img, img_metas, **kwargs):
        super().forward_train(img, img_metas, **kwargs)
        kwargs.update({"img": img})
        kwargs.update({"img_metas": img_metas})
        kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
        data_groups = dict_split(kwargs, "tag")
        for _, v in data_groups.items():
            v.pop("tag")

        loss = {}
        # ! Warnings: By splitting losses for supervised data and unsupervised data with different names,
        # ! it means that at least one sample for each group should be provided on each gpu.
        # ! In some situation, we can only put one image per gpu, we have to return the sum of loss
        # ! and log the loss with logger instead. Or it will try to sync tensors don't exist.
        if "sup" in data_groups:
            gt_bboxes = data_groups["sup"]["gt_bboxes"]
            log_every_n(
                {"sup_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            sup_loss = self.student.forward_train(**data_groups["sup"])
            sup_loss = {"sup_" + k: v for k, v in sup_loss.items()}
            loss.update(**sup_loss)
        if "unsup_student" in data_groups:
            unsup_loss = weighted_loss(
                self.forward_unsup_train(
                    data_groups["unsup_teacher"], data_groups["unsup_student"]
                ),
                weight=self.unsup_weight,
            )
            unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}
            loss.update(**unsup_loss)

        return loss

    def forward_unsup_train(self,teacher_data,student_data):
        #学生信息的title为teacherdata中的img——meta，解析字典kv对再封装成列表list，
        tnames = [meta["filename"] for meta in teacher_data["img_metas"]]
        snames = [meta["filename"] for meta in student_data["img_metas"]]
        tidx = [tnames.index(name) for name in snames]
        with torch.no_grad():
            teacher_info = self.extract_teacher_info(
                    teacher_data["img"][
                        torch.Tensor(tidx).to(teacher_data["img"].device).long()
                    ],
                    [teacher_data["img_metas"][idx] for idx in tidx],
                    [teacher_data["proposals"][idx] for idx in tidx]
                    if ("proposals" in teacher_data)
                       and (teacher_data["proposals"] is not None)
                    else None,
            )
        student_info = self.extract_student_info(**student_data)

        return self.compute_pseudo_label_loss(student_info, teacher_info)

    def compute_pseudo_label_loss(self, student_info, teacher_info):
        M = self._get_trans_mat(
            teacher_info["transform_matrix"], student_info["transform_matrix"]
        )

        pseudo_bboxes = self._transform_bbox(
            teacher_info["det_bboxes"],
            M,
            [meta["img_shape"] for meta in student_info["img_metas"]]
        )
        pseudo_labels = teacher_info["det_labels"]

        loss = {}

        rpn_loss, proposal_list = self.rpn_loss(
            student_info["rpn_out"],
            pseudo_bboxes,
            student_info["img_metas"],
            student_info = student_info,
        )
        loss.update(rpn_loss)

        "this proposal_list is student's rpn out "
        if proposal_list is not None:
            student_info["proposals"] = proposal_list
        if self.train_cfg.use_teacher_proposal:
            proposals = self._transform_bbox(
                teacher_info["proposals"],
                M,
                [meta["img_shape"] for meta in student_info["img_metas"]],
            )
        else:
            proposals = student_info["proposals"]

        loss.update(
            self.unsup_rcnn_cls_loss(
                student_info["backbone_feature"],
                student_info["img_metas"],
                proposals,
                pseudo_bboxes,
                pseudo_labels,
                student_info=student_info,
            )
        )
        loss.update(
            self.unsup_rcnn_reg_loss(
                student_info["backbone_feature"],
                student_info["img_metas"],
                proposal_list,
                pseudo_bboxes,
                pseudo_labels,
                student_info=student_info,
                )
            )
        return loss
    def rpn_loss(
            self,
            rpn_out,
            pseudo_bboxes,
            img_metas,
            gt_bboxes_ignore = None,
            student_info = None,
            **kwargs
        ):
        if self.student.with_rpn:
            gt_bboxes = []
            for bbox in pseudo_bboxes:
                bbox, _ , _ = filter_invalid(
                    bbox[:,:4],
                    score = bbox[:,4],
                    thr = self.train_cfg.rpn_pseudo_threshold,
                    min_size = self.train_cfg.min_pseduo_box_size,
                )
                gt_bboxes.append(bbox)
            log_every_n(
                {"rpn_groundtrues_num":sum([len(box) for box in gt_bboxes])/len(gt_bboxes)}
            )
            loss_inputs = rpn_out + [[bbox.float() for bbox in gt_bboxes], img_metas]
            losses = self.student.rpn_head.loss(
                *loss_inputs, gt_bboxes_ignore = gt_bboxes_ignore
            )
            proposal_cfg = self.student.train_cfg.get(
                "rpn_proposal", self.student.test_cfg.rpn
            )
            proposal_list = self.student.rpn_head.get_bboxes(
                *rpn_out,img_metas = img_metas,cfg = proposal_cfg
            )
            log_image_with_boxes(
                "rpn",
                student_info["img"][0],
                pseudo_bboxes[0][:,:4],
                bbox_tag = "rpn_pseudo_label",
                scores = pseudo_bboxes[0][:, 4],
                interval = 500,
                img_norm_cfg = student_info["img_metas"][0]["img_norm_cfg"],
            )
            return losses, proposal_list
        else:
            return {} ,None
    def unsup_rcnn_cls_loss(
            self,
            feat,
            img_metas,
            proposal_list,
            pseudo_bboxes,
            pseudo_labels,
            student_info=None,
            **kwargs,
    ):
        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [bbox[:, 4] for bbox in pseudo_bboxes],
            thr=self.train_cfg.cls_pseudo_threshold,
        )
        log_every_n(
            {"rcnn_cls_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        sampling_results = self.get_sampling_result(
            img_metas,
            proposal_list,
            gt_bboxes,
            gt_labels,
        )
        selected_bboxes = [res.bboxes[:, :4] for res in sampling_results]
        rois = bbox2roi(selected_bboxes)


        bbox_results = self.student.roi_head._bbox_forward(feat, rois)
        roi_feat = bbox_results["bbox_feats"]
        bbox_targets = self.student.roi_head.bbox_head.get_targets(
         sampling_results, gt_bboxes, gt_labels, self.student.train_cfg.rcnn
        )

        samples_relation_score = self.student.roi_head.bbox_head.student_gcn(roi_feat).squeeze()

        "bbox_results拿着去求学生网络的GCN预测分数，用于求loss时reweighting加权"
        "先获得背景roi的索引"
        assigned_label, _, _, _ = bbox_targets
        neg_indexs = assigned_label == self.student.roi_head.bbox_head.num_classes
        bbox_targets[1][neg_indexs] = samples_relation_score[neg_indexs]
        log_every_n({"samples_relation_score": sum(samples_relation_score) / neg_indexs.numel()})
        log_every_n({"neg_samples_relation_score": sum(samples_relation_score[neg_indexs]) / neg_indexs.nonzero(as_tuple=False).numel()})
        loss = self.student.roi_head.bbox_head.loss(
            bbox_results["cls_score"],
            bbox_results["bbox_pred"],
            rois,
            *bbox_targets,
            reduction_override="none"
        )
        "这里的操作是分类loss求和再归一化，如果所有proposal的分类权重大于1的话，需要除掉该权重"
        if 'loss_cls' in loss.keys():

            loss['loss_cls'] = loss['loss_cls'].sum() / max(bbox_targets[1].sum(), 1.0)
        "这里的操作是回归loss求和再归一化，如果目标框的个数大于1，要除以目标框数"
        loss['loss_bbox'] = loss['loss_bbox'].sum() / max(
            bbox_targets[1].size()[0], 1.0
        )
        if len(gt_bboxes[0]) > 0:
            log_image_with_boxes(
                "rcnn_cls",
                student_info["img"][0],
                gt_bboxes[0],
                bbox_tag="pseudo_label",
                labels=gt_labels[0],
                class_names=self.CLASSES,
                interval=500,
                img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            )
        return loss

    def unsup_rcnn_reg_loss(self,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        student_info = None,
        **kwargs,
    ):
        "refinescores是图卷积网络对教师网络roihead的proposal之间特征关系的建模，是proposal间的关系分数，对于预测分数更高的proposal"
        "与这些proposal的特征关系分数更高的但预测分数较低的proposal"
        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [bbox[:, -1] for bbox in pseudo_bboxes],
            thr= self.train_cfg.relation_score_thr,
        )
        log_every_n(
            {"rcnn_reg_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        loss_bbox = self.student.roi_head.forward_train(
            feat, img_metas, proposal_list, gt_bboxes, gt_labels, **kwargs
        )["loss_bbox"]
        if len(gt_bboxes[0]) > 0:
            log_image_with_boxes(
                "rcnn_reg",
                student_info["img"][0],
                gt_bboxes[0],
                bbox_tag="pseudo_label",
                labels=gt_labels[0],
                class_names=self.CLASSES,
                interval=500,
                img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            )
        return {"loss_bbox": loss_bbox}
    def get_sampling_result(
        self,
        img_metas,
        proposal_list,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        **kwargs,
    ):
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.student.roi_head.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
            )
            sampling_result = self.student.roi_head.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
            )
            sampling_results.append(sampling_result)
        return sampling_results

    @force_fp32(apply_to=["bboxes", "trans_mat"])
    def _transform_bbox(self, bboxes, trans_mat, max_shape):
        bboxes = Transform2D.transform_bboxes(bboxes, trans_mat, max_shape)
        return bboxes

    @force_fp32(apply_to=["a", "b"])
    def _get_trans_mat(self, a, b):
        return [bt @ at.inverse() for bt, at in zip(b, a)]




    def extract_teacher_info(self,img, img_metas, proposals=None, **kwargs):
        teacher_info = {}
        feat = self.teacher.extract_feat(img)
        teacher_info["backbone_feature"] = feat
        if proposals is None:
            proposal_cfg  = self.teacher.train_cfg.get(
                "rpn_proposal",self.teacher.test_cfg.rpn
            )
            rpn_out = list(self.teacher.rpn_head(feat))
            proposal_list = self.teacher.rpn_head.get_bboxes(
                *rpn_out,img_metas = img_metas,cfg = proposal_cfg
            )
        else:
            proposal_list = proposals
        teacher_info["proposals"] = proposal_list


        ##rpn's proposal_list传入ROIhead,out is  a batch's det_bboxes and det_labels
        ##det_bboxes is a list of batchs'detbboxes of tensor(N,6) the last dim is GCN's out of relation_score
        det_bboxes, det_labels = self.teacher.roi_head.simple_test_bboxes_with_gcn(feat, img_metas, proposal_list, self.teacher.test_cfg.rcnn, rescale=False)

        ####
        proposal_list = det_bboxes
        proposal_label_list = det_labels


        proposal_list = [p.to(feat[0].device) for p in proposal_list]
        proposal_list = [p if p.shape[0] > 0 else p.new_zeros(0, 6) for p in proposal_list]
        proposal_label_list = [p.to(feat[0].device) for p in proposal_label_list]


        if isinstance(self.train_cfg.pseudo_label_initial_score_thr, float):
            thr = self.train_cfg.pseudo_label_initial_score_thr
        else:
            # TODO: use dynamic threshold
            raise NotImplementedError("Dynamic Threshold is not implemented yet.")
        proposal_list, proposal_label_list, _ = list(
            zip(
                *[
                    filter_invalid(
                        proposal,
                        proposal_label,
                        proposal[:, -2],
                        thr=thr,
                        min_size=self.train_cfg.min_pseduo_box_size,
                    )
                    for proposal, proposal_label in zip(
                        proposal_list, proposal_label_list
                    )
                ]
            )
        )
        det_bboxes = proposal_list
        det_labels = proposal_label_list
        teacher_info["det_bboxes"] = det_bboxes
        teacher_info["det_labels"] = det_labels
        teacher_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        teacher_info["img_metas"] = img_metas
        return teacher_info

    def extract_student_info(self,img, img_metas, proposals=None, **kwargs):
        student_info = {}
        student_info["img"] = img
        feat = self.student.extract_feat(img)
        student_info["backbone_feature"] = feat
        if self.student.with_rpn:
            rpn_out = self.student.rpn_head(feat)
            student_info["rpn_out"] = list(rpn_out)
        student_info["img_metas"] = img_metas
        student_info["proposals"] = proposals
        student_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        return student_info


    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if not any(["student" in key or "teacher" in key for key in state_dict.keys()]):
            keys = list(state_dict.keys())
            state_dict.update({"teacher." + k: state_dict[k] for k in keys})
            state_dict.update({"student." + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

