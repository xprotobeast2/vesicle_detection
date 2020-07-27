import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class DetectorLossFn(nn.Module):
    def __init__(self):
        super().__init__()
        # self.ce_fn = nn.CrossEntropyLoss(reduction='none')
        # self.mse_fn = nn.MSELoss(reduction='none')

    def single_target_bbox_ious(self, pred_boxes, target_box):
        '''
            pred_boxes: idx, (x, y, w, h)
            target_box: (tlx, tly, brx, bry)
        '''
        box2 = target_box.unsqueeze(0).expand(*pred_boxes.shape)
        box1 = pred_boxes

        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2] + box1[:, 0], box1[:, 3] + box1[:, 1]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # get the corrdinates of the intersection rectangle
        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)
        # Intersection area
        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
            inter_rect_y2 - inter_rect_y1 + 1, min=0
        )
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou

    def forward(self, pred_boxes, pred_cls, target, cls_weight=1.0, x_weight=1.0, y_weight=1.0, w_weight=1.0, h_weight=1.0, conf_weight=1.0):
        '''
            pred_boxes: (N, K, 5) tensor
            pred_cls: (N, K, n_classes) tensor
            target: (N, maxObj, class + 4) tensors
        '''
        N = pred_boxes.shape[0]
        max_objects = target.shape[1]
        n_classes = pred_cls.shape[2]

        target_mask = (target.sum(2, keepdim=True) != 0)
        target_mask_fl = target_mask.float()

        target_boxes = target[..., 1:]  # Top-left x, y and bottom-right x, y
        target_cls = target[..., 0].long()

        pred_conf = pred_boxes[..., 4]
        pred_boxes = pred_boxes[..., :4]

        gather_indices_boxes = torch.zeros(*target_boxes.shape).long().to(target_boxes.device)
        gather_indices_cls = torch.zeros((N, max_objects, n_classes)).long().to(target_boxes.device)
        gather_indices_conf = torch.zeros((N, max_objects)).long().to(target_boxes.device)
        for b in range(N):
            for t in range(target[b].shape[0]):
                if not target_mask[b, t, 0]:
                    continue
                ious = self.single_target_bbox_ious(pred_boxes[b], target_boxes[b, t])
                best_box = torch.argmax(ious)
                gather_indices_boxes[b, t, :] = best_box
                gather_indices_cls[b, t, :] = best_box

        best_pred_boxes = torch.gather(pred_boxes, 1, gather_indices_boxes)
        best_pred_cls = torch.gather(pred_cls, 1, gather_indices_cls)
        best_pred_conf = F.sigmoid(torch.gather(pred_conf, 1, gather_indices_conf))

        loss_dict = {}
        # Class loss
        loss_cls = F.cross_entropy(best_pred_cls.view(-1, n_classes), target_cls.view(-1), reduction='none')[target_mask.view(-1)].mean()
        loss_dict['loss_cls'] = loss_cls.item()
        # mse loss
        loss_x = F.mse_loss(best_pred_boxes[..., 0].view(-1), target_boxes[..., 0].view(-1), reduction='none')[target_mask.view(-1)].mean()
        loss_dict['loss_x'] = loss_x.item()
        loss_y = F.mse_loss(best_pred_boxes[..., 1].view(-1), target_boxes[..., 1].view(-1), reduction='none')[target_mask.view(-1)].mean()
        loss_dict['loss_y'] = loss_y.item()
        loss_w = F.mse_loss(best_pred_boxes[..., 2].view(-1), (target_boxes[..., 2] - target_boxes[..., 0]).view(-1), reduction='none')[target_mask.view(-1)].mean()
        loss_dict['loss_w'] = loss_w.item()
        loss_h = F.mse_loss(best_pred_boxes[..., 3].view(-1), (target_boxes[..., 3] - target_boxes[..., 1]).view(-1), reduction='none')[target_mask.view(-1)].mean()
        loss_dict['loss_h'] = loss_h.item()

        conf_labels = (best_pred_conf > 0.5).float()
        loss_conf = F.binary_cross_entropy(best_pred_conf.view(-1), conf_labels.view(-1), reduction='none')[target_mask.view(-1)].mean()
        loss_dict['loss_conf'] = loss_conf.item()

        loss = (
            cls_weight * loss_cls
            + x_weight * loss_x
            + y_weight * loss_y
            + w_weight * loss_w
            + h_weight * loss_h
            + conf_weight * loss_conf
        )
        return loss, loss_dict


class DiceLoss(nn.Module):
    def __init__(self, class_weights=None):
        super(DiceLoss, self).__init__()
        self.w = class_weights

    def forward(self, preds, targets):
        intersection = torch.einsum('bcij, bcij -> bc', [preds, targets])
        union = torch.einsum('bcij, bcij -> bc', [preds, preds]) + \
                torch.einsum('bcij, bcij -> bc', [targets, targets])

        iou = torch.div(intersection, union)
        dice = torch.div(2*intersection, union)

        avg_dice = (torch.einsum('bc->c', dice) / (targets.shape[0]))
                
        if self.w is not None:    
            avg_dice = torch.dot(self.w, avg_dice)/targets.shape[1]
        else:
            avg_dice = torch.einsum('c->', avg_dice)/targets.shape[1]

        return (1 - avg_dice)



