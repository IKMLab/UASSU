import numpy as np
import torch
import torch.nn.functional as F

import math


class Generator:
    def __init__(self):
        super().__init__()
        self.support_decoding_methods = ['top-k']
        return

    @staticmethod
    def top_k(
        model_output: torch.Tensor, 
        k: int, 
        ):

        if not isinstance(model_output, torch.Tensor):
            raise TypeError(f'Expect parameter `model_output`: `{model_output}` to be type `torch.Tensor`.')

        if not isinstance(k, int):
            raise TypeError(f'Expect parameter `k`: `{k}` to be type `int`.')

        top_k_output, top_k_indexes = torch.topk(model_output, k)

        # choose 1 index in each instance with given k_prob
        batch_top_k_indexes = torch.multinomial(top_k_output, 1, replacement=True)

        # get vocabulary indexes from top-k indexes
        batch_token_ids = torch.gather(top_k_indexes, 1, batch_top_k_indexes)

        return batch_token_ids.flatten().tolist()

