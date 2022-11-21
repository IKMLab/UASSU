import torch
import torch.nn as nn

import sys

class ReconstructionLoss(nn.Module):
    """
    Implementation of cross entropy loss with optional label smoothing & customized weighted loss.
    """

    def __init__(self, pad_idx):
        """
        Args:
            - `pad_idx`       (int)
        """

        if not isinstance(pad_idx, int):
            raise TypeError(f'Expect parameter `pad_idx`: `{pad_idx}` to be type `int`.')

        super(ReconstructionLoss, self).__init__()

        self.pad_idx = pad_idx
        self.logsoftmax = nn.LogSoftmax(dim=1)

        return

    def forward(self, prediction, target, masking_pos=None, masked_loss_weight=None):
        """
        Args:
            - `prediction`     (Tensor) : (B x S x V) # .view(-1, V)
            - `target`         (Tensor) : (B x S)
            - `masking_pos`    (Tensor) : (B x S) 1s' are masked areas, 0s' are non-masked areas.
            - `masked_loss_weight` (float)  : N, non-masking weight:masking weight = (1-N):N  # (B x S).view(-1)
        """

        batch_size = target.shape[0]
        seq_len = target.shape[1]
        vocab_size = prediction.shape[2]

        invalid_loss = False

        if torch.any(torch.isnan(prediction)):
            print('ISNAN!!! - prediction:')
            print(prediction)
            invalid_loss = True
        if torch.any(~torch.isfinite(prediction)):
            print('ISINFINITE!!! - prediction:')
            print(prediction)
            invalid_loss = True
        if torch.any(torch.isnan(target)):
            print('ISNAN!!! - target:')
            print(target)
            invalid_loss = True
        if torch.any(~torch.isfinite(target)):
            print('ISINFINITE!!! - target:')
            print(target)
            invalid_loss = True
        
        if invalid_loss: sys.exit(0)

        prediction = prediction.reshape(-1, prediction.size(-1))
        target = target.flatten()

        logsoftmax_prediction = self.logsoftmax(prediction)
        logsoftmax_prediction = logsoftmax_prediction.view(-1, vocab_size) # BS x V
        one_hot = torch.zeros_like(logsoftmax_prediction).scatter(1, target.view(-1, 1), 1)

        loss = -(one_hot * logsoftmax_prediction) # element-wise multiplication

        # if use `masked_loss_weight` & `masking_pos`.
        if masked_loss_weight is not None and masking_pos is not None:
            # make weights
            complement_masking_pos = torch.ones_like(masking_pos) - masking_pos # 0s' are masked areas, 1s' are non-masked areas.
            masked_loss_weight = masked_loss_weight * masking_pos + (1-masked_loss_weight) * complement_masking_pos
            masked_loss_weight = masked_loss_weight.view(-1, 1)
            masked_loss_weight_sum = masked_loss_weight.sum()
            # apply weights
            loss = masked_loss_weight * loss / masked_loss_weight_sum
        
        loss = loss.sum(dim=1)

        # ignore paddings
        non_pad_mask = target.ne(self.pad_idx).view(-1)
        loss = loss.masked_select(non_pad_mask).sum()

        loss /= batch_size
        loss /= seq_len

        return loss
