import torch
import torch.nn.functional as F

def focal_loss(preds, labels, alpha=0.25, gamma=2.0, reduction='mean'):
    """
    Focal Loss for addressing class imbalance
    
    Args:
        preds: predictions (logits)
        labels: ground truth labels (0 or 1)
        alpha: weighting factor for rare class
        gamma: focusing parameter
    """
    # Convert logits to probabilities
    p = torch.sigmoid(preds)
    
    # Calculate focal loss
    ce_loss = F.binary_cross_entropy_with_logits(preds, labels, reduction='none')
    p_t = p * labels + (1 - p) * (1 - labels)
    alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
    focal_weight = alpha_t * (1 - p_t) ** gamma
    
    focal_loss = focal_weight * ce_loss
    
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss

def masif_site_focal_loss(preds, labels, alpha=0.25, gamma=2.0):
    """
    Improved MaSIF loss using focal loss instead of random subsampling
    """
    # Just use focal loss on all points - no subsampling needed
    loss = focal_loss(preds, labels, alpha=alpha, gamma=gamma)
    return loss, preds, labels

def weighted_masif_site_loss(preds, labels):
    """
    Weighted loss that gives more weight to positive examples
    """
    pos_weight = (labels == 0).sum().float() / (labels == 1).sum().float()
    loss = F.binary_cross_entropy_with_logits(preds, labels, 
                                              pos_weight=pos_weight)
    return loss, preds, labels 