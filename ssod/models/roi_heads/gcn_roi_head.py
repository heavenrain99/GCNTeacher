import torch
from torch import nn
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead
from mmdet.core import bbox2roi
from ssod.utils import log_image_with_boxes, log_every_n
@HEADS.register_module()
class GCNRoIHead(StandardRoIHead):
    """
    RoI head for GCNTeacher
    with StrandardROIhead GCNRoiHead need output proposals relation score with input roi_feat.
    if GCNHead 's input is teacher's det_bbox(100,5), edges is build from every det_box IOU relation
    eles  input is student's proposals(512,5), edges is build from every proposals CosineSimilirity Dis
    """

    def simple_test_bboxes_with_gcn(self, x,
                                         img_metas,
                                         proposals,
                                         rcnn_test_cfg,
                                         rescale=False):
        """

        Args:
            x: backbone_feat
            img_metas:
            proposals:
            rcnn_test_cfg:
            rescale:

        Returns:
a
        """
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

        bbox_results = self._bbox_forward(x, rois)
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
            bbox_pred = (None,) * len(proposals)

        det_bboxes = []
        det_labels = []
        roi_relation_scores = []

        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                det_bbox = rois[i].new_zeors(0, 6)
                det_label = rois[i].new_zeros((0,), dtype=torch.long)
                if rcnn_test_cfg is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros(0, self.bbox_head.fc_cls.out_features)
                roi_relation_score = rois[i].new_zeros(0, 64)
            else:
                det_bbox, det_label, det_inds = self.bbox_head.get_bboxes_with_inds(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=False,
                    cfg=rcnn_test_cfg
                )

                det_inds = torch.floor(det_inds / 80).long()

                det_feat = roi_feat[i][det_inds, ...]

                roi_relation_score = self.bbox_head.teacher_gcn(det_feat, det_bbox)
                if det_feat.shape[0] > 0 :
                    log_every_n({"teacher_relation_score": sum(roi_relation_score) / det_feat.shape[0]})

            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
            roi_relation_scores.append(roi_relation_score)
        det_bboxes = [
            torch.cat([bbox, score], dim=-1) for bbox, score in zip(det_bboxes, roi_relation_scores)
        ]
        return det_bboxes, det_labels
