import torch
import torch.nn.functional as F


class SupConLoss(torch.nn.Module):

    def __init__(self, temperature=0.5, contrast_mode='all',
                 base_temperature=0.07, scale_by_temperature=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, x1, x2,labels=None, dev_score=None, mask=None):
        x1 = F.normalize(x1, p=2, dim=1)
        x2 = F.normalize(x2, p=2, dim=1)
        features = torch.cat([x1, x2], dim=1)

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        ##############
        dev_score = torch.where(dev_score > 5.0, torch.ones_like(dev_score)*5.0, dev_score)
        dev_score = torch.where(dev_score < -5.0, torch.ones_like(dev_score)*(-5.0), dev_score)
        dev_score = dev_score.view(1, -1)
        dev_score_b = (dev_score-dev_score.T).abs()
        mask = torch.where(dev_score_b.abs() <= 1, torch.ones_like(dev_score_b), torch.zeros_like(dev_score_b))
        mask = mask.float().to(device)
        ##############

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
        

        # tile mask
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask

        ################
        sim_coeff = 1.0/(1.0+dev_score_b)
        ################

        num_positives_per_row = torch.sum(positives_mask, axis=1)  
        denominator = torch.sum(
            exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
            exp_logits * positives_mask, axis=1, keepdims=True)

        log_probs = (logits - torch.log(denominator + 1e-6))*sim_coeff
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        log_probs = torch.sum(
            log_probs * positives_mask, axis=1)[num_positives_per_row > 0] / num_positives_per_row[
                        num_positives_per_row > 0]
        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        # loss = loss.mean()
        return loss


