import torch
from mmcv.runner.fp16_utils import force_fp32
from mmdet.core import bbox2roi, multi_apply, bbox_overlaps
from mmdet.models import DETECTORS, build_detector
from torch import nn
from ssod.utils.structure_utils import dict_split, weighted_loss
from ssod.utils import log_image_with_boxes, log_every_n
from torch.nn.modules.utils import _pair
from .multi_stream_detector import MultiSteamDetector
from .utils import Transform2D, filter_invalid

@DETECTORS.register_module()
class GCNTeacher(MultiSteamDetector):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super(GCNTeacher, self).__init__(
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
        bbox_targets = self.student.roi_head.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.student.train_cfg.rcnn
        )

        roi_feat = bbox_results["bbox_feats"]
        refine_score = self.Student_Semantic_GCN(roi_feat)
        refine_score = torch.sum(refine_score,dim=1)
        "bbox_results拿着去求学生网络的GCN预测分数，用于求loss时reweighting加权"
        "先获得背景roi的索引"
        assigned_label, _, _, _ = bbox_targets
        neg_indexs = assigned_label == self.student.roi_head.bbox_head.num_classes
        bbox_targets[1][neg_indexs] = refine_score[neg_indexs].detach()

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
        det_bboxes, det_labels = self.teacher_roi_simple_test_with_GCN(feat,img_metas,proposal_list,self.teacher.test_cfg.rcnn, rescale=False)

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
    def teacher_roi_simple_test_with_GCN(self,x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        rois = bbox2roi(proposals)
        if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_bbox = rois.new_zeros(0, 6)
            det_label = rois.new_zeros((0,), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :4]
                det_label = rois.new_zeros(
                    (0, self.bbox_head.fc_cls.out_features))
            # There is no proposal in the whole batch
            return [det_bbox] * batch_size, [det_label] * batch_size


        bbox_results = self.teacher.roi_head._bbox_forward(x, rois)
        img_shapes = tuple(meta["img_shape"] for meta in img_metas)
        scale_factors = tuple(meta["scale_factor"] for meta in img_metas)

        roi_feat = bbox_results["bbox_feats"]
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        num_proposals_per_img = tuple(len(p) for p in proposals)

        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        roi_feat = roi_feat.split(num_proposals_per_img, 0)
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)




        det_bboxes = []
        det_labels = []
        roi_relation_scores = []

        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                det_bbox = rois[i].new_zeors(0, 6)
                det_label = rois[i].new_zeros((0,), dtype=torch.long)
                if self.teacher.test_cfg.rcnn is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros(0, self.teacher.roi_head.bbox_head.fc_cls.out_features)
                roi_relation_score = rois[i].new_zeros(0, 64)
            else:
                det_bbox, det_label, det_inds = self.teacher.roi_head.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=False,
                    cfg=self.teacher.test_cfg.rcnn
                )

                det_inds = torch.floor(det_inds / 80).long()

                det_feat = roi_feat[i][det_inds, ...]
                roi_relation_score = self.Teacher_Context_GCN(det_bbox, det_feat)

            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
            roi_relation_scores.append(roi_relation_score)
        det_bboxes = [
            torch.cat([bbox, score], dim=-1) for bbox, score in zip(det_bboxes, roi_relation_scores)
        ]
        return det_bboxes, det_labels
    def featmapPooling(self,x):
        pooling = nn.Sequential(
            nn.AvgPool2d(self.roi_feat_size),
        )
        pooling.to(x.device)
        return pooling(x)
    def W1(self,x):
        w1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        w1.to(x.device)
        return w1(x)
    def W2(self,x):
        w2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        w2.to(x.device)
        return w2(x)
    def Mlp(self, x):
        mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        mlp.to(x.device)
        #torch.save(mlp.state_dict(), '/home/xys/Desktop/SoftTeacher-main/2.pth')

        return mlp(x)
    def Teacher_Context_GCN(self,det_box,roi_feat):
        if det_box.shape[0]==0 or roi_feat.shape[0]==0:
            return det_box.new_zeros(0,)
        representation_vector = torch.squeeze(self.featmapPooling(roi_feat))
        representation_vector = self.Mlp(representation_vector)
        adjacency_matrix = self.build_edges_of_Context_graph(det_box[:,:4])
        D = torch.diag(torch.sum(adjacency_matrix, dim=1) ** -0.5)
        A = D @ adjacency_matrix @ D

        if representation_vector.ndim == 1 :
            representation_vector = representation_vector.unsqueeze(0)
        relation_score = torch.softmax(torch.unsqueeze(torch.sum(self.W2(A @ self.W1(A @ representation_vector)), dim=-1), -1),dim=0)
        return relation_score






    def build_edges_of_Context_graph(self,proposal):
        """

        Args:
            proposal_list: ROIhead's output , tensor with shape of (N,5)

        Returns:
            adjacency_matrix represent location relations between proposals
            by each proposal 's iou between others
            tensor with shape of (N,N)
        """

        proposal = torch.unsqueeze(proposal, dim=0)
        adjacency_matrix = bbox_overlaps(proposal, proposal)
        adjacency_matrix = torch.squeeze(adjacency_matrix, dim=0)
        return adjacency_matrix

    #学生网络语义关系推理
    def Student_Semantic_GCN(self, roi_feat):
        if(roi_feat.shape[0] == 0):
            return roi_feat.new_zeros(0,64)
        "特征图压缩成向量"
        X1 = self.featmapPooling(roi_feat)
        X1 = X1.squeeze()

        if X1.ndim == 1:
            X1 = X1.unsqueeze(0)
        feat_vec = self.Mlp(X1)
        "依据特征向量间的余弦距离建立roi特征间的语义关系矩阵"
        adjacency_matrix = self.cosine_similarity(feat_vec,feat_vec)

        D = torch.diag(torch.sum(adjacency_matrix, dim=1) ** -0.5)
        A = D @ adjacency_matrix @ D

        relation_score = torch.softmax(torch.unsqueeze(torch.sum(self.W2(A @ self.W1(A @ feat_vec)), dim=-1), -1),dim=0)
        return relation_score





    def cosine_similarity(self, x, y):
        # x: m x d
        # y: n x d
        # return: m x n
        assert x.size(1) == y.size(1)
        x = torch.nn.functional.normalize(x, dim=1)
        y = torch.nn.functional.normalize(y, dim=1)
        x = x.unsqueeze(1).expand(x.size(0), y.size(0), x.size(1))  # [m,1*n,d]
        y = y.unsqueeze(0).expand(x.shape)  # [1*m,n,d]
        return (x * y).sum(2)

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
