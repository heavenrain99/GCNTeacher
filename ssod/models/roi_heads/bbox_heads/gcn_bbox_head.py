
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmdet.core import bbox2roi, multi_apply, bbox_overlaps
from mmdet.core import build_bbox_coder, multi_apply, multiclass_nms
from mmdet.models.builder import HEADS, build_loss
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmdet.models.utils import build_linear_layer
from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import Shared2FCBBoxHead


@HEADS.register_module()
class GCNBBoxHead(Shared2FCBBoxHead):
    """
    rewrite GCNROIHEAD 's GCNBBOXHEAD 's get_Bboxs method ,need to return det_inds(keeps) of det_bbox after nms
    and GCNet
    """

    def __init__(self,init_cfg=None,
                 *args,
                 **kwargs,
                 ):
        super(GCNBBoxHead, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        self.pooling = nn.AvgPool2d(self.roi_feat_size)

        self.fc_in = build_linear_layer(dict(type='Linear'), 256, 128)
        self.graph_fc1 = build_linear_layer(dict(type='Linear'), 128, 128)
        self.relu = nn.ReLU(inplace=True)
        self.graph_fc2 = build_linear_layer(dict(type='Linear'), 128, 64)
        self.fc_out = build_linear_layer(dict(type='Linear'), 64, 1)
        self.sigmoid = nn.Sigmoid()
        _,self.Batchnorm2d = build_norm_layer(dict(type = 'BN'),256)
        _,self.Batchnorm1d = build_norm_layer(dict(type = "BN1d"),1)
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

    def build_teacher_edges(self, x):
        x = x[:,:4]
        proposal = torch.unsqueeze(x, dim=0)
        adjacency_matrix = bbox_overlaps(proposal, proposal)
        adjacency_matrix = torch.squeeze(adjacency_matrix, dim=0)
        return adjacency_matrix


    def build_student_edges(self,x):
        return self.cosine_similarity(x, x)

    @force_fp32(apply_to='x')
    def graph_convolution_network(self, x, edges):

        if x.shape[0]==0 or edges.shape[0]==0:
            return x.new_zeros(0,)
        D = torch.diag(torch.sum(edges, dim=1) ** -0.5)
        A = D @ edges @ D
        if x.ndim == 1 :
            x = x.unsqueeze(0)
        y1 = self.relu(self.graph_fc1(A @ x))
        y2 = self.relu(self.graph_fc2(A @ y1))
        y2 = torch.where(torch.isnan(y2),torch.full_like(y2,0),y2)
        y2 = torch.where(torch.isinf(y2),torch.full_like(y2,0),y2)
        y2 = self.fc_out(y2)
        y2 = torch.where(torch.isnan(y2), torch.full_like(y2, 0), y2)
        y2 = torch.where(torch.isinf(y2), torch.full_like(y2, 0), y2)
        if y2.shape[0]>1:
            y2 = self.Batchnorm1d(y2)
        scores = self.sigmoid(y2)
        scores = torch.where(torch.isnan(scores), torch.full_like(scores, 1), scores)
        scores = torch.where(torch.isinf(scores), torch.full_like(scores, 1), scores)
        return scores


    def student_gcn(self,roi_feat):
        if roi_feat.shape[0] == 0:
            return roi_feat.new_zeors(0,)
        roi_feat = self.Batchnorm2d(roi_feat)
        feat_vec = torch.squeeze(self.pooling(roi_feat))
        feat_vec = self.relu(self.fc_in(feat_vec))
        edges = self.build_student_edges(feat_vec)
        return self.graph_convolution_network(feat_vec, edges)


    def teacher_gcn(self,det_feat,det_bbox):
        if det_bbox.shape[0]==0 or det_feat.shape[0]==0:
            return det_bbox.new_zeros(0,)
        det_feat = self.Batchnorm2d(det_feat)
        feat_vec = torch.squeeze(self.pooling(det_feat))
        feat_vec = self.relu(self.fc_in(feat_vec))
        edges = self.build_teacher_edges(det_bbox)
        return self.graph_convolution_network(feat_vec, edges)




    # def _get_target_single_with_gcn(self,feat,pos_bboxes, neg_bboxes, pos_gt_bboxes,
    #                        pos_gt_labels, cfg):
    #
    #
    #     ##change label_weights of neg from gcn's output relationscore
    #     num_pos = pos_bboxes.size(0)
    #     num_neg = neg_bboxes.size(0)
    #     num_samples = num_pos + num_neg
    #
    #     labels = pos_bboxes.new_full((num_samples,),
    #                                  self.num_classes,
    #                                  dtype=torch.long)
    #     label_weights = pos_bboxes.new_zeros(num_samples)
    #     bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
    #     bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
    #     edges = self.build_student_edges(feat)
    #     samples_relation_score = self.graph_convolution_network(feat, edges)
    #     if num_pos > 0:
    #         labels[:num_pos] = pos_gt_labels
    #         pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
    #         label_weights[:num_pos] = pos_weight
    #         if not self.reg_decoded_bbox:
    #             pos_bbox_targets = self.bbox_coder.encode(
    #                 pos_bboxes, pos_gt_bboxes)
    #         else:
    #             # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
    #             # is applied directly on the decoded bounding boxes, both
    #             # the predicted boxes and regression targets should be with
    #             # absolute coordinate format.
    #             pos_bbox_targets = pos_gt_bboxes
    #         bbox_targets[:num_pos, :] = pos_bbox_targets
    #         bbox_weights[:num_pos, :] = 1
    #     if num_neg > 0:
    #         label_weights[-num_neg:] = samples_relation_score[-num_neg:]
    #
    #     return labels, label_weights, bbox_targets, bbox_weights
    #
    # def get_targets_with_gcn(self,
    #                 feat,
    #                 sampling_results,
    #                 gt_bboxes,
    #                 gt_labels,
    #                 rcnn_train_cfg,
    #                 concat=True):
    #     pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
    #     neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
    #     pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
    #     pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
    #     labels, label_weights, bbox_targets, bbox_weights = multi_apply(
    #         self._get_target_single_with_gcn,
    #         feat,
    #         pos_bboxes_list,
    #         neg_bboxes_list,
    #         pos_gt_bboxes_list,
    #         pos_gt_labels_list,
    #         cfg=rcnn_train_cfg)
    #
    #     if concat:
    #         labels = torch.cat(labels, 0)
    #         label_weights = torch.cat(label_weights, 0)
    #         bbox_targets = torch.cat(bbox_targets, 0)
    #         bbox_weights = torch.cat(bbox_weights, 0)
    #     return labels, label_weights, bbox_targets, bbox_weights

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes_with_inds(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None
                   ):
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
                bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels, keeps = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img, return_inds=True)

            return det_bboxes, det_labels,keeps