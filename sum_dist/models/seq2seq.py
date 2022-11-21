import torch
import torch.nn as nn
from transformers import BertTokenizer

import random
import sys

from sum_dist.models.decoder import TransformerDecoder


class Seq2seqModel(nn.Module):
    def __init__(
        self, 
        word_embed_size, 
        vocab_size, 
        encoder, 
        encoder_out_embed_size,
        decoder_in_embed_size,
        window_size, 
        slide_step, 
        decoder, 
        device, 
        logger=None,
        span_aggregation_choice='avg', 
        masking_ratio=0.15,
        masking_weight=1.0):

        super(Seq2seqModel, self).__init__()
        self.logger = logger

        self.encoder = encoder
        self.decoder = decoder

        # encoder & decoder share embeddings
        self.decoder.embeddings = self.encoder.get_embedding_layer()

        self.encoder_out_linear = None
        if span_aggregation_choice == 'cat':
            self.encoder_out_linear = nn.Linear(encoder_out_embed_size, decoder_in_embed_size)

        self.device = device

        self.window_size = window_size
        self.slide_step = slide_step
        self.span_aggregation_choice = span_aggregation_choice
        self.masking_ratio = masking_ratio
        self.masking_weight = masking_weight
        return


    def forward(
        self, 
        encoder_input_ids, 
        encoder_attention_mask, 
        encoder_token_type_ids, 
        tgt, 
        tgt_mask, 
        tgt_key_padding_mask,
        print_detail=False):

        """
        Get input(BatchSize x Seq Length), forward through encoder -> window -> masking -> decoder.

        Returns:
            - `out`: normal forwarding output (forward through transformer encoder/span aggregation/masking/decoder)
            - `masked_ids` (Tensor) : (B x S)
            - `cur_span_reprs`: reprs without masking
            - `cur_spans`: tokens corresponding to span reprs
        """

        if print_detail:
            self.logger.info(f'In seq2seq.py: Start converting input type.')

        encoder_input_ids = encoder_input_ids.to(self.device)
        encoder_attention_mask = encoder_attention_mask.to(self.device)
        encoder_token_type_ids = encoder_token_type_ids.to(self.device)
        tgt = tgt.to(self.device)
        tgt_mask = tgt_mask.to(self.device)
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = tgt_key_padding_mask.to(self.device)

        if print_detail:
            self.logger.info(f'In seq2seq.py: Start encoder.')

        out = self.encoder(
            input_ids=encoder_input_ids, 
            attention_mask=encoder_attention_mask, 
            token_type_ids=encoder_token_type_ids)

        if print_detail:
            self.logger.info(f'In seq2seq.py: Start span aggregation.')

        out, memory_key_padding_mask, spans = self.span_aggregation(
            out, 
            input_ids=encoder_input_ids, 
            attention_mask=encoder_attention_mask)
        cur_span_reprs = out
        cur_spans = spans

        out = out.transpose(0, 1)
        
        if print_detail:
            self.logger.info(f'In seq2seq.py: Start generate mask.')

        memory_mask, masked_ids = self.masking_strategy(
            span_ids=spans,
            target_max_len=tgt.shape[0],
            num_spans=out.shape[0],
            print_detail=print_detail
        )

        if print_detail:
            self.logger.info('Memory mask:')
            self.logger.info(memory_mask)
            if memory_mask is not None:
                self.logger.info(memory_mask.shape)
                self.logger.info(memory_mask[0])
            self.logger.info('Masked ids:')
            self.logger.info(masked_ids)
            if masked_ids is not None:
                self.logger.info(masked_ids.shape)
                self.logger.info(masked_ids[0])
            self.logger.info('Decoder tgt:')
            self.logger.info(tgt)
            self.logger.info(tgt.shape)
            self.logger.info('Encoder out:')
            self.logger.info(out)
            self.logger.info(out.shape)
            self.logger.info('Decoder tgt mask:')
            self.logger.info(tgt_mask)
            self.logger.info(tgt_mask.shape)
            self.logger.info('Decoder tgt pad mask:')
            self.logger.info(tgt_key_padding_mask)
            self.logger.info(tgt_key_padding_mask.shape)
            self.logger.info('Memory key pad mask:')
            self.logger.info(memory_key_padding_mask)
            self.logger.info(memory_key_padding_mask.shape)
            self.logger.info(torch.sum(memory_key_padding_mask, 1))
            self.logger.info(f'In seq2seq.py: Start decoder.')

        out = self.decoder(
            tgt=tgt, 
            span_reprs=out, 
            tgt_mask=tgt_mask, 
            memory_mask=memory_mask, 
            tgt_key_padding_mask=tgt_key_padding_mask, 
            memory_key_padding_mask=memory_key_padding_mask)

        return out, masked_ids, cur_span_reprs, cur_spans


    def span_aggregation(self, reprs, attention_mask, input_ids=None):
        """
        Decide how to form a span representation.

        Args:
            `reprs`          (Tensor) : (BatchSize x SeqLength x ReprEmbedSize)
            `attention_mask` (Tensor) : (B x S)
            `input_ids`      (Tensor) : (B x S) For creating `spans`.

        Returns:
            `window_reprs`  (Tensor)          : (BatchSize x (SeqLen-WindowSize+1) x ReprSize) under self.span_aggregation_choice == `avg`
            `padding_masks` (Tensor)          : masks for decoding input (memory mask)
            `spans`         (List[List[str]]) : corresponding tokens of each window span per batch
        """
        if self.span_aggregation_choice == 'avg':
            """
            Sum the beginning repr and the endding repr of each window with given slide step.
            E.g. [A B C D E F G] with window 5 and slide step 1, returns [(A+E), (B+F), (C+G)] as `window_reprs`.
            """
            # creating reprs
            window_reprs = []

            for index in range(0, reprs.shape[1]-self.window_size+self.slide_step, self.slide_step):
                end_index = (index + self.window_size - 1)
                if end_index > reprs.shape[1] - 1:
                    end_index = reprs.shape[1] - 1

                indices = torch.tensor( [ index, end_index ] ).to(self.device)
                window_reprs.append(torch.sum(torch.index_select(reprs, 1, indices), dim=1))
            
            if len(window_reprs) == 0:
                window_reprs = torch.sum(reprs, dim=1)
                window_reprs = torch.unsqueeze(window_reprs, 1)
            else:
                try:
                    window_reprs = torch.stack(window_reprs, dim=1)
                except:
                    print(f'Error~~~~~')
                    print(f'Empty Seq in seq2seq.py span_aggregation(). !!!!!!!!')
                    print('window_reprs:')
                    print(window_reprs.shape)
                    sys.exit(0)

        elif self.span_aggregation_choice == 'cat':
            """
            Concat the beginning and the endding repr in each window.
            E.g. [A B C D E F G] with window 5 and slide step 1, returns [[A,E], [B,F], [C,G]] as `window_reprs`.
            """
            # creating reprs
            window_reprs = []

            for index in range(0, reprs.shape[1]-self.window_size+self.slide_step, self.slide_step):
                end_index = (index + self.window_size - 1)
                if end_index > reprs.shape[1] - 1:
                    end_index = reprs.shape[1] - 1

                begin_indices = torch.tensor( [ index ] ).to(self.device)
                end_indices = torch.tensor( [ end_index ] ).to(self.device)
                begin_reprs = torch.index_select(reprs, 1, begin_indices)
                end_reprs = torch.index_select(reprs, 1, end_indices)
                merged_reprs = torch.cat((begin_reprs, end_reprs), dim=2)

                window_reprs.append(merged_reprs)
            
            if len(window_reprs) == 0:
                print(f'Error~~~~~')
                print(f'Empty Seq in seq2seq.py span_aggregation(). !!!!!!!!')
                print('window_reprs:')
                print(window_reprs.shape)
                sys.exit(0)

            window_reprs = torch.stack(window_reprs, dim=1)
            window_reprs = window_reprs.squeeze()
            if len(window_reprs.shape) == 2:
                window_reprs = window_reprs.unsqueeze(0)
            window_reprs = self.encoder_out_linear(window_reprs)

        
        elif self.span_aggregation_choice == 'all_avg':
            """
            Sum the beginning repr and the endding repr of each window with given slide step.
            E.g. [A B C D E F G] with window 5 and slide step 1, returns [mean(A+B+C+D+E), mean(B+C+D+E+F), mean(C+D+E+F+G)] as `window_reprs`.
            """
            # creating reprs
            indices = torch.arange(0, reprs.shape[1]).to(self.device)
            traversal = indices.unfold(
                dimension=0, size=self.window_size, step=self.slide_step
            )
            window_reprs = [
                torch.mean(torch.index_select(reprs, 1, window_i), dim=1) for window_i in traversal
            ]

            if len(window_reprs) == 0:
                window_reprs = torch.mean(reprs, dim=1)
                window_reprs = torch.unsqueeze(window_reprs, 1)
            else:
                try:
                    window_reprs = torch.stack(window_reprs, dim=1)
                except:
                    print(f'Error~~~~~')
                    print(f'Empty Seq in seq2seq.py span_aggregation(). !!!!!!!!')
                    print('window_reprs:')
                    print(window_reprs.shape)
                    sys.exit(0)

        # creating memory key pad masks
        batch_num_reprs = attention_mask.shape[1] - torch.sum(attention_mask.float(), dim=1) # B, `False` for unmasked, `True` for masked

        batch_num_new_attn = torch.ceil((batch_num_reprs - self.window_size) / self.slide_step + 1)

        padding_masks = torch.zeros(window_reprs.shape[0], window_reprs.shape[1])
        for ind, num_attn in enumerate(batch_num_new_attn):
            padding_masks[ind][num_attn.int():] = 1
        padding_masks = padding_masks.bool().to(self.device)

        # creating corresponding span tokens
        spans = None
        if input_ids is not None:
            spans = [
                instance_ids.unfold(0, 5, 1).tolist() for instance_ids in input_ids
            ]
        
        return window_reprs, padding_masks, spans


    def masking_strategy(self, span_ids=None, print_detail=False, **kwargs):
        """
        Decide how to generate masks of span representations for the decoder.

        Args:
            `span_ids` (List[List[str]]) : (B x S(len according to each instance)) corresponding tokens of each window span per batch

        Returns:
            `masks` (BoolTensor or FloatTensor): (TargetLen x SourceLen)
            `masked_ids` (Tensor): (B x S(len according to each instance)) True for masked ids & False for unmasked ids.
        """
        if print_detail:
            self.logger.info('Create base masks.')

        """
        If window size is 5, generate: `0111101111011...` or `1101111011110111...` (0s' are separated by `window_size-1` 1s').
        positions with 1s' can not be attended (masks).
        positions with 0s' where corresponding values will be unchanged.
        """
        # create base masks (BoolTensor)
        start_index = random.randint(0, self.window_size-1)
        indices = torch.tensor(
            [1 if i % self.window_size != start_index else 0 for i in range(0, kwargs['num_spans'])]
        ).to(self.device).bool()
        masks = indices.repeat(kwargs['target_max_len'], 1)
        # control mask ratio
        if print_detail:
            self.logger.info('Apply mask ratio.')

        if self.masking_ratio > 0:
            rand_table = torch.rand(masks.shape).to(self.device) < self.masking_ratio
            masks = (masks + rand_table) > 0

        # get masked span token ids (tokens corresponding to 1s')
        if print_detail:
            self.logger.info('Get masked span token ids.')

        masked_ids = None
        if span_ids is not None:
            target_masked_ids = torch.ones(kwargs['target_max_len'], dtype=torch.long)
            for index, mask_value in enumerate(masks[0]):
                start_ind = index*self.slide_step
                if start_ind >= kwargs['target_max_len']:
                    break
                if mask_value == False:
                    target_masked_ids[start_ind:start_ind+self.window_size] = 0

            masked_ids = target_masked_ids.repeat(len(span_ids), 1)

        # control mask weight
        if print_detail:
            self.logger.info('Apply mask weight.')

        if self.masking_weight != 1:
            # convert to FloatTensor, which would be added to attn weight
            masks = masks * (-self.masking_weight)

        return masks, masked_ids.to(self.device)


    def encoder_inference(
        self, 
        encoder_input_ids, 
        encoder_attention_mask, 
        encoder_token_type_ids, 
        tgt_key_padding_mask):

        encoder_input_ids = encoder_input_ids.to(self.device)
        encoder_attention_mask = encoder_attention_mask.to(self.device)
        encoder_token_type_ids = encoder_token_type_ids.to(self.device)
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = tgt_key_padding_mask.to(self.device)

        out = self.encoder(
            input_ids=encoder_input_ids, 
            attention_mask=encoder_attention_mask, 
            token_type_ids=encoder_token_type_ids)

        out, memory_key_padding_mask, spans = self.span_aggregation(
            out, 
            input_ids=encoder_input_ids, 
            attention_mask=encoder_attention_mask)

        out = out.transpose(0, 1)
        
        return out, memory_key_padding_mask, spans


    def decoder_inference(
        self,
        tgt,
        span_reprs,
        tgt_mask,
        tgt_key_padding_mask,
        memory_key_padding_mask,
        memory_mask=None,
        ):

        """
        Go through decoder once.
        # (No) Apply attention filtering (as soft masks on `span_reprs`) in each iteration.

        Used in `observe_preliminary.py`, `generator.py`.
        """

        out = self.decoder(
            tgt=tgt, 
            span_reprs=span_reprs, 
            tgt_mask=tgt_mask, 
            memory_mask=memory_mask, 
            tgt_key_padding_mask=tgt_key_padding_mask, 
            memory_key_padding_mask=memory_key_padding_mask)

        return out # B x S x V
