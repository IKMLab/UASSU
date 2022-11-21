import numpy as np
from sentence_transformers import SentenceTransformer, util
import spacy
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import os
import sys

from sum_dist.generator import Generator


@torch.no_grad()
def beam_search(
    model,
    input_sequence,
    bos_id,
    eos_id,
    beam_width: int,
    device: torch.device,
    max_seq_len: int,
):

    batch_size = input_sequence.size(0)
    accum_prob = [0] * (batch_size * beam_width)
    # outputs = torch.LongTensor([[bos_id] * max_seq_len] * batch_size).to(device)
    outputs = torch.zeros(batch_size, max_seq_len, dtype=torch.long).to(device)
    outputs[:, 0] = bos_id
    # log which beam is selected
    parents = [[] for _ in range(batch_size)]
    for curr_len in range(1, max_seq_len):
        pred = model(input_sequence, outputs)
        pred = torch.nn.functional.softmax(pred, dim=-1)
        probs, indices = pred[:, curr_len-1, :].cpu().topk(k=beam_width, dim=-1)
        # probs, indices = pred[:, -1, :].cpu().topk(k=5, dim=-1)
        # indices = indices.take(probs.multinomial(1))
        probs = probs.log()
        if curr_len == 1:
            accum_prob = probs.view(-1)
            input_sequence = input_sequence.repeat(1, beam_width, 1) # B, M*S, H (M=Beam Width)
            input_sequence = input_sequence.view(batch_size * beam_width, -1, input_sequence.size(-1)) # B*M , S,H
            outputs = torch.LongTensor([[bos_id] * max_seq_len] * batch_size * beam_width ).to(device)
            outputs[:, curr_len] = indices.view(-1)
            continue

        all_beams = []
        for batchidx, (batch_out, batch_accu_prob, batch_probs, batch_indices) in enumerate(zip(
                outputs.view(batch_size, beam_width, -1),
                accum_prob.view(batch_size, -1),
                probs.view(batch_size, -1),
                indices.view(batch_size, -1),
        )):
            beams = []
            for beamidx, (beam_out, beam_acc_prob, beam_probs, beam_indices) in enumerate(zip(
                    batch_out,
                    batch_accu_prob,
                    batch_probs.view(beam_width, -1),
                    batch_indices.view(beam_width, -1))):
                beams.extend([( beam_out, [int(beam_index)], beam_acc_prob + beam_prob, beamidx) for beam_index, beam_prob in zip(
                    beam_indices,
                    beam_probs)])
            # print(beams)
            beams = sorted(beams, key=lambda x: x[2], reverse=True)[:beam_width]
            # print(beams)
            # exit()
            parents[batchidx].append(tuple(zip(*beams))[3])
            all_beams.extend(beams)

        outs, next_token, accum_prob, _ = list(zip(*all_beams))
        outputs = torch.stack(outs, 0)
        outputs[:, curr_len] = torch.LongTensor(next_token).view(-1)
        accum_prob = torch.stack(accum_prob)

    return outputs.view(batch_size, beam_width, -1)[:,0,:], torch.LongTensor(parents)[:,:,0]


class SpanFreqTool(object):
    def __init__(self):
        self.sbert = SentenceTransformer('paraphrase-distilroberta-base-v2')
        return

    def get_embeds(self, sents):
        try:
            embeds = self.sbert.encode(sents, convert_to_tensor=True)
        except:
            return None
        return embeds


class AttnHook():
    def __init__(self):
        self.handles = []
        self.attn_weight_list = []
        self.cur_len = 0
        return

    def attn_hook_fn(self, module, layer_in, layer_out):
        """
        See PyTorch Doc - MultiheadAttention:
            layer_out[0] for attn out, layer_out[1] for attn weight.
        """
        # get attention weight
        cur_attn_weight = layer_out[1] # attn weight, B x DecoderMaxLen x EncoderOut
        # get attn of cur decoding step
        cur_attn_weight = cur_attn_weight[:, self.cur_len, :] # B x EncoderOut
        cur_attn_weight = cur_attn_weight.detach().cpu().numpy()
        self.attn_weight_list.append(cur_attn_weight.copy())

    def register(self, module):
        self.handles.append(module.register_forward_hook(self.attn_hook_fn))
        return

    def deregister(self):
        for handle in self.handles:
            handle.remove()
        self.attn_weight_list = []
        return


class Trainer(object):

    def __init__(
        self, 
        config, 
        model, 
        tokenizer,
        rouge_calculator,
        device, 
        log_dir,
        logger=None,
        optimizer=None,
        ):

        super(Trainer, self).__init__()

        self.logger = logger
        self.writer = SummaryWriter(log_dir=log_dir)
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.rouge_calculator = rouge_calculator
        self.device = device
        self.global_step = 0


    def train(
        self, 
        epoch_num, 
        loss_fn, 
        data_loader_train,
        collate_fn,
        accumulation_step, 
        save_checkpoint_step, 
        save_checkpoint_dir,
        prediction_dest,
        gold_dest,
        cur_step=0, 
        start_batch_idx=0, 
        mode='normal',
        print_detail=False):

        """
        Args:
            - `epoch_num`            (int)
            - `loss_fn`
            - `accumulation_step`    (int)
            - `save_checkpoint_step` (int)
            - `cur_step`             (int)       : current global step
            - `start_batch_idx`      (int)       : skip until the specific batch index, then start to train
            - `mode`                 (str)       : ['normal', 'order']
                - `normal`  : normal training
                - `order`   : only logging procedure orders & run 2 batches only
            - `print_detail`         (bool)
        """
        # initialization
        print_log = print_detail
        if mode == 'order': print_log = True

        if print_log:
            self.logger.info('In trainer.py - train()')

        if self.optimizer is None:
            raise NotImplementedError('Parameter `optimizer` in Class Trainer not initiated.')

        self.model.train()
        base_step = 0
        self.global_step = cur_step
        cur_loss_weight = self.config.loss_weight_levels[self.config.cur_loss_weight_level]

        self.logger.info(f'In trainer initialization:')
        self.print_current_status({
            'cur_base_step': base_step,
            'cur_loss_weight': cur_loss_weight,
            'cur_sampling_len': collate_fn.encoder_sampling_len,
        })

        # train
        for epoch_index in tqdm(range(epoch_num)):
            if mode == 'order':
                self.logger.info(f'Epoch {epoch_index}')

            weighted_accumulation_loss = 0

            self.optimizer.zero_grad()

            # ===== TRAIN =====
            for batch_index, batch in enumerate(tqdm(data_loader_train)):
                decoder_tgt = batch['decoder_tgt'].to(self.device)
                if mode == 'order':
                    self.logger.info(f'Batch {batch_index}')
                    self.logger.info('Batch input:')
                    self.logger.info(batch)

                if batch_index < start_batch_idx and epoch_index == 0:
                    continue

                encoder_attention_mask = batch['attention_mask'] == 0 # float to bool, 1s' for not masked, 0s' for masked positions

                output, masking_pos, _, _ = self.model(
                    encoder_input_ids=batch['input_ids'].to(self.device),
                    encoder_attention_mask=encoder_attention_mask.to(self.device),
                    encoder_token_type_ids=batch['token_type_ids'].to(self.device),
                    tgt=decoder_tgt, 
                    tgt_mask=batch['decoder_tgt_mask'].to(self.device), 
                    tgt_key_padding_mask=batch['decoder_tgt_key_padding_mask'].to(self.device),
                    print_detail=print_log)
                
                if mode == 'order':
                    self.logger.info(f'Model forward done.')
                    self.logger.info('Model output:')
                    self.logger.info(output[:, :decoder_tgt.shape[0]-1])
                    self.logger.info(output[:, :decoder_tgt.shape[0]-1].shape)
                    self.logger.info('Model output masking pos:')
                    self.logger.info(masking_pos[:, 1:])
                    self.logger.info(masking_pos[:, 1:].shape)
                    self.logger.info('Model target:')
                    self.logger.info(decoder_tgt.transpose(0, 1)[:, 1:])
                    self.logger.info(decoder_tgt.transpose(0, 1)[:, 1:].shape)
                
                # loss
                if masking_pos is not None:
                    masking_pos = masking_pos[:, 1:]

                batch_loss = loss_fn(
                    prediction=output[:, :decoder_tgt.shape[0]-1], 
                    target=decoder_tgt.transpose(0, 1)[:, 1:], # ignore the first token `[CLS]`
                    masking_pos=masking_pos,
                    masked_loss_weight=cur_loss_weight)

                # backward & accumulate
                batch_loss.backward()
                weighted_accumulation_loss += batch_loss.item()/accumulation_step

                if print_log:
                    self.logger.info(f'Batch loss: {batch_loss}')
                    self.logger.info(f'Loss calculating done.')

                # update
                if base_step % accumulation_step == 0 and base_step > 0:
                    if print_log:
                        self.logger.info(f'Start backward.')

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    self.save_record({
                            'Train/WeightedLoss': weighted_accumulation_loss,
                            'Config/lr': self.config.lr,
                            'Config/sampling_length': collate_fn.encoder_sampling_len,
                            'Config/masking_ratio': self.model.masking_ratio,
                            'Config/masking_weight': self.model.masking_weight,
                            'Config/loss_weight': cur_loss_weight,
                        },
                        step=self.global_step)

                    # free
                    weighted_accumulation_loss = 0

                    if print_log:
                        self.logger.info(f'Backward done.')

                # save checkpoint & records
                if (base_step % (save_checkpoint_step*accumulation_step) == 0 and base_step > 0) or mode == 'order':
                    self.print_current_status({
                        'cur_base_step': base_step,
                        'cur_loss_weight': cur_loss_weight,
                        'cur_sampling_len': collate_fn.encoder_sampling_len,
                    })

                    self.save_checkpoint(save_checkpoint_dir, print_log)
                
                # batch end: free
                if base_step % accumulation_step == 0 and base_step > 0:
                    self.global_step += 1

                base_step += 1

                # output.detach()
                # batch_loss.detach()

                # torch.cuda.empty_cache()

                if mode == 'order':
                    self.logger.info('Batch done.')
                    if batch_index == 1:
                        break

            # epoch end
            if mode == 'order':
                break

        # final save
        self.logger.info(f'In trainer saving-checkpoint:')
        self.print_current_status({
            'cur_base_step': base_step,
            'cur_loss_weight': cur_loss_weight,
            'cur_sampling_len': collate_fn.encoder_sampling_len,
        })

        if mode != 'order':
            self.save_checkpoint(save_checkpoint_dir, print_log)

        # training end
        self.writer.flush()
        self.writer.close()

    def generate_sequence_beam(self, batch, encoder_output, memory_mask,
                        decoder_tgt_key_padding_mask, memory_key_padding_mask,
                        bos_id, max_len, final_decode_len):
        batch_size = batch['input_ids'].shape[0]
        src_mask_beamx = None if memory_key_padding_mask is None else memory_key_padding_mask.repeat(self.config.beam_size, 1)

        if decoder_tgt_key_padding_mask is not None:
            raise Exception('why?')
        maxlen = min(final_decode_len,max_len)
        tgt_sub = batch['decoder_tgt_mask'][:maxlen,:maxlen].to(self.device)
        # Create a wrapper instead of modify my beam search
        r, p = beam_search(
            model =lambda src, tgt : self.model.decoder_inference(
                tgt=tgt.transpose(0, 1),
                span_reprs=src.transpose(0, 1),
                tgt_mask=tgt_sub,
                memory_mask=memory_mask,
                tgt_key_padding_mask=None,
                memory_key_padding_mask = memory_key_padding_mask if (src.size(0) == batch_size) else src_mask_beamx,
            )  # B*beam_width x S x V
            ,input_sequence= encoder_output.transpose(0, 1),
            bos_id=bos_id,
            eos_id=-1,
            beam_width=self.config.beam_size,
            device=self.device,
            max_seq_len=final_decode_len
        )
        return r.transpose(0, 1), p


    def inference(
        self, 
        data_loader, 
        batch_size,
        max_len=500, 
        final_decode_len=50,
        decode_times=2,
        return_attn=False,
        use_high_rouge=False, # use sents w/ high ROUGE to decode
        use_high_freq=False, # use sents w/ high freq to decode
        similarity_threshold=0.5,
        spacy_model_ver=None,
        filter_len=200,
        ):

        """
        Call once and get inference results of the given `data_loader` using top-k decoding.
        Inference results should contain specified elements: decoding results or attention in decoder.

        Args:
            - `data_loader`
            - `max_len`      (int)       : decoding max length
            - `decode_times` (int)       : Run through decoder N times. If N > 1, then apply attention filtering after decoding.

        Returns:
            - `results`       (List[str]): all decoding results, each include `[CLS]`
                - len: batch size
            - `attn_results`  (List[List[np.array]]): all decoding attn
                - len: hook size
                    - len: data size
                        - len: seq len
            - `encoder_spans` (List[List[List[str]]]): all input spans
                - len: data size
                    - len: seq_len
                        - len: window size
        """
        assert sum([use_high_rouge, use_high_freq]) <= 1

        if use_high_freq:
            freq_tool = SpanFreqTool()

        self.model.eval()
        sent_tokenizer = spacy.load(spacy_model_ver)

        if return_attn:
            self.create_and_register_hook()

        bos_id = self.tokenizer.convert_tokens_to_ids(self.config.bos_token)

        decoding_results = []
        attn_results = []
        encoder_spans_for_attn = []

        def beam_att(att_list, parents):
            parents= np.array(parents)
            # att Memory order : GenLen-1, Batch*Beam, SrcLen
            _att = np.stack(att_list[1:])
            # att Memory order : GenLen-1, Batch, Beam, SrcLen -> Batch, GenLen-1, Beam, SrcLen
            _att = _att.reshape(_att.shape[0], len(parents), -1, _att.shape[2]).transpose(1, 0, 2, 3)
            # each selected beam has a parent indicate which beam it came from
            # -> Batch, GenLen-1, 1, SrcLen
            _att = np.take_along_axis(_att, parents.reshape(*parents.shape, 1, 1), 2)
            # output Batch, GenLen-1, SrcLen -> (Batch, SrcLen) + (Batch, SrcLen) -> (Batch, SrcLen)
            return _att.reshape(len(parents), -1, _att.shape[-1]).sum(1) + att_list[0]

        for batch_index, batch in enumerate(tqdm(data_loader)):
            # filter bad data when decoding twice (batch size == 1)
            if len(batch['articles']) == 1 and len(batch['articles'][0]) < 50 and decode_times > 1:
                continue

            """ encode once """

            decoder_tgt_key_padding_mask = None
            if batch['decoder_tgt_key_padding_mask'] is not None:
                decoder_tgt_key_padding_mask = batch['decoder_tgt_key_padding_mask'].to(self.device)

            encoder_attention_mask = batch['attention_mask'] == 0 # float to bool

            encoder_output, memory_key_padding_mask, encoder_span_tokens = self.model.encoder_inference(
                encoder_input_ids=batch['input_ids'].to(self.device),
                encoder_attention_mask=encoder_attention_mask.to(self.device),
                encoder_token_type_ids=batch['token_type_ids'].to(self.device),
                tgt_key_padding_mask=decoder_tgt_key_padding_mask) # B x S x V

            """ decode the first time """
            if len(self.config.decoding_methods) > 1:
                raise ValueError("Only one decoding method can be specified.")
            if 'top-k' in self.config.decoding_methods:
                cur_generated_sequences = self.generate_sequence(
                    batch=batch, 
                    encoder_output=encoder_output, 
                    memory_mask=None, 
                    decoder_tgt_key_padding_mask=decoder_tgt_key_padding_mask, 
                    memory_key_padding_mask=memory_key_padding_mask, 
                    bos_id=bos_id,
                    max_len=max_len,
                    final_decode_len=final_decode_len)  
            elif 'beam' in self.config.decoding_methods:
                cur_generated_sequences, parents= self.generate_sequence_beam(
                    batch=batch, 
                    encoder_output=encoder_output, 
                    memory_mask=None, 
                    decoder_tgt_key_padding_mask=decoder_tgt_key_padding_mask, 
                    memory_key_padding_mask=memory_key_padding_mask, 
                    bos_id=bos_id,
                    max_len=max_len,
                    final_decode_len=final_decode_len)
            else:
                raise ValueError("Only top-k or beam can be specified as `decoding_methods`.")

            """ decode the second time """

            if decode_times > 1:
                # divide sents
                sent_spans = sent_tokenizer(batch['articles_sampled'][0])
                sent_lens = [
                    len(self.tokenizer(
                        text=sent_span.text, 
                        add_special_tokens=False, 
                        return_token_type_ids=None, 
                        return_attention_mask=None)['input_ids'])
                    for sent_span in sent_spans.sents]

                # apply layer-0 attn filtering
                if 'top-k' in self.config.decoding_methods:
                    batch_attn_result = np.stack(self.attn_hooks[0].attn_weight_list.copy()).sum(axis=0)
                elif 'beam' in self.config.decoding_methods:
                    batch_attn_result = beam_att(self.attn_hooks[0].attn_weight_list, parents)
                    
                # scale memory_mask (>0 for larger attn, <0 for smaller attn)
                unit_memory_mask = batch_attn_result / 100
                memory_mask = np.stack([unit_memory_mask for _ in range(cur_generated_sequences.shape[0])], axis=1).squeeze()
                memory_mask = torch.from_numpy(memory_mask).to(self.device)

                if use_high_freq:
                    # get embeds
                    cur_embeds = freq_tool.get_embeds([' '.join(self.tokenizer.convert_ids_to_tokens(est)) for est in encoder_span_tokens[0]]) # S x E
                    # get indices of top sbert freq spans
                    cosine_scores = util.pytorch_cos_sim(cur_embeds, cur_embeds) # S x S
                    freq = cosine_scores > similarity_threshold
                    freq_list = torch.sum(freq, dim=0)
                    freq_list = freq_list - 1 # S
                    freq_list = freq_list.detach().cpu().numpy() # S

                    # select sents with highest freq
                    sent_weights = []
                    sent_token_idx_range = []
                    cur_start_token_idx = 0
                    for token_num_in_sent in sent_lens:
                        end_index = cur_start_token_idx + token_num_in_sent
                        if end_index > batch_attn_result.shape[1]:
                            end_index = batch_attn_result.shape[1]

                        if end_index > cur_start_token_idx:
                            sent_weights.append(sum(freq_list[cur_start_token_idx:end_index])/(end_index-cur_start_token_idx))
                        else:
                            sent_weights.append(0)
                        sent_token_idx_range.append((cur_start_token_idx, end_index))
                        cur_start_token_idx += token_num_in_sent

                elif use_high_rouge:
                    # get rouge
                    self.rouge_calculator.convert_article_to_rouge_file([batch['highlights'][0]], gold_dir='./temp/sent_target', is_prediction=False)

                    # select sents with highest rouge-2
                    sent_weights = []
                    sent_token_idx_range = []
                    cur_start_token_idx = 0
                    for token_num_in_sent in sent_lens:
                        end_index = cur_start_token_idx + token_num_in_sent
                        if end_index > batch_attn_result.shape[1]:
                            end_index = batch_attn_result.shape[1]

                        if end_index > cur_start_token_idx:
                            cur_sent = self.tokenizer.decode(batch['input_ids'][0][1+cur_start_token_idx:1+end_index]) # start from [cls]
                            self.rouge_calculator.convert_article_to_rouge_file(
                                [cur_sent], 
                                is_prediction=True, 
                                prediction_dir=f'./temp/sent_pred')

                            scores = self.rouge_calculator.get_score(
                                prediction_dir=f'./temp/sent_pred',
                                gold_dir='./temp/sent_target')
                            sent_weights.append(scores['rouge_2_f_score'])
                        else:
                            sent_weights.append(0)
                        sent_token_idx_range.append((cur_start_token_idx, end_index))
                        cur_start_token_idx += token_num_in_sent

                else:
                    # method-3 SENT SOFT MASKS: reorder sent by attn, finally order by appear order
                    # select sents with highest attn weights
                    sent_weights = []
                    sent_token_idx_range = []
                    cur_start_token_idx = 0
                    for token_num_in_sent in sent_lens:
                        end_index = cur_start_token_idx + token_num_in_sent
                        if end_index > batch_attn_result.shape[1]:
                            end_index = batch_attn_result.shape[1]

                        if end_index > cur_start_token_idx:
                            sent_weights.append(sum(batch_attn_result[0][cur_start_token_idx:end_index])/(end_index-cur_start_token_idx))
                        else:
                            sent_weights.append(0)
                        sent_token_idx_range.append((cur_start_token_idx, end_index))
                        cur_start_token_idx += token_num_in_sent


                # concatenate batch_attn of high weighted sentences as filter_indexes
                assert len(sent_lens) == len(sent_token_idx_range)
                sent_index_sorted = np.argsort(sent_weights)[::-1].copy()
                filter_sent_indexes = []
                cur_token_num = 0
                for sent_index in sent_index_sorted:
                    if cur_token_num > filter_len: break
                    if sent_token_idx_range[sent_index][1] > sent_token_idx_range[sent_index][0]:
                        filter_sent_indexes.append(sent_index)
                        cur_token_num += sent_lens[sent_index]

                # filter input
                filter_sent_indexes = sorted(filter_sent_indexes)
                filter_indexes = [torch.arange(sent_token_idx_range[sent_index][0], sent_token_idx_range[sent_index][1]) for sent_index in filter_sent_indexes]
                filter_indexes = torch.cat(filter_indexes).to(self.device)

                # apply filtered input
                encoder_output = torch.index_select(encoder_output, 0, filter_indexes)
                memory_mask = torch.index_select(memory_mask, 1, filter_indexes)
                memory_key_padding_mask = None
                encoder_spans_for_attn += [[encoder_span_tokens[0][i] for i in filter_indexes]]

                # clear batch attn weight
                for hook in self.attn_hooks:
                    hook.attn_weight_list = []

                # decode
                if 'top-k' in self.config.decoding_methods:
                    cur_generated_sequences = self.generate_sequence(
                        batch=batch, 
                        encoder_output=encoder_output, 
                        memory_mask=memory_mask, 
                        decoder_tgt_key_padding_mask=decoder_tgt_key_padding_mask, 
                        memory_key_padding_mask=memory_key_padding_mask, 
                        bos_id=bos_id,
                        max_len=max_len,
                        final_decode_len=final_decode_len)
                elif 'beam' in self.config.decoding_methods:
                    cur_generated_sequences, parents = self.generate_sequence_beam(
                        batch=batch, 
                        encoder_output=encoder_output, 
                        memory_mask=memory_mask, 
                        decoder_tgt_key_padding_mask=decoder_tgt_key_padding_mask, 
                        memory_key_padding_mask=memory_key_padding_mask, 
                        bos_id=bos_id,
                        max_len=max_len,
                        final_decode_len=final_decode_len)

            """ all decoding end here """

            decoding_results.append(cur_generated_sequences.detach().cpu().numpy())

            if return_attn: # will only record attn from the last decoding time
                tmp_attn_result = []
                for hook_ind, hook in enumerate(self.attn_hooks):
                    if 'top-k' in self.config.decoding_methods:
                        tmp_attn_result.append(np.stack(hook.attn_weight_list.copy()).sum(axis=0))
                    elif 'beam' in self.config.decoding_methods:
                        tmp_attn_result.append(beam_att(hook.attn_weight_list, parents))
                
                attn_results.append(np.array(tmp_attn_result.copy()))

                # clear batch attn weight
                for hook in self.attn_hooks:
                    hook.attn_weight_list = []

        # convert to output format (Dict[List[str]])
        final_results = []
        for instance_result in decoding_results:
            instance_result = np.array(instance_result)
            instance_result = np.transpose(instance_result, (1, 0)) # B x S
            if 'top-k' in self.config.decoding_methods:
                assert instance_result.shape[1] == max_len
                instance_result = np.reshape(instance_result, (-1, max_len)).tolist()
            elif 'beam' in self.config.decoding_methods:
                assert instance_result.shape[1] ==final_decode_len
                instance_result = np.reshape(instance_result, (-1, final_decode_len)).tolist()
            final_results += [self.tokenizer.decode(instance_ids) for instance_ids in instance_result]

        self.writer.flush()

        if return_attn:
            for hook in self.attn_hooks:
                hook.deregister()

            # convert `attn_results` output format
            tmp_method_val_attn_results = []

            hook_size = len(attn_results[0])
            for hook_ind in range(hook_size):
                tmp_hook_val_attn_results = []
                for batch_attn_result in attn_results:
                    for instance_attn_result in batch_attn_result[hook_ind]:
                        tmp_hook_val_attn_results.append(instance_attn_result.copy())
                tmp_method_val_attn_results.append(tmp_hook_val_attn_results.copy())

            attn_results = tmp_method_val_attn_results.copy() # should be HookSize x DataSize x EncoderInputSize

            # convert span ids to span tokens for saving attn viz img
            encoder_span_tokens_for_attn = []
            for encoder_span in encoder_spans_for_attn:
                tmp_instance_span_tokens = []
                for spans in encoder_span:
                    tmp_instance_span_tokens.append(self.tokenizer.convert_ids_to_tokens(spans))
                encoder_span_tokens_for_attn.append(tmp_instance_span_tokens.copy())

            return final_results, attn_results, encoder_span_tokens_for_attn
        else:
            return final_results, None, None


    def generate_sequence(
            self, 
            batch, 
            encoder_output, 
            memory_mask, 
            decoder_tgt_key_padding_mask, 
            memory_key_padding_mask, 
            bos_id,
            max_len,
            final_decode_len):

        cur_batch_num = batch['input_ids'].shape[0]
        cur_len = 1
        cur_generated_ids = torch.LongTensor([[bos_id for _ in range(cur_batch_num)]]+[[0 for _ in range(cur_batch_num)]]*(max_len-cur_len)).to(self.device)

        while cur_len < max_len:
            # update tgt
            tgt = cur_generated_ids

            # decode
            output = self.model.decoder_inference(
                tgt=tgt,
                span_reprs=encoder_output,
                tgt_mask=batch['decoder_tgt_mask'].to(self.device),
                memory_mask=memory_mask, # apply last layer attn value from last decoding
                tgt_key_padding_mask=decoder_tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask) # B x S x V

            # get the latest output
            output = output.detach()
            output = torch.index_select(output, 1, torch.LongTensor([cur_len-1]).to(self.device))
            output = output.squeeze()
            if len(output.shape) == 1:
                output = F.softmax(output/self.config.t, dim=0)
                output = output.unsqueeze(dim=0)
            else:
                output = F.softmax(output/self.config.t, dim=1)

            # call different decoding methods
            new_generated_ids = Generator.top_k(output, self.config.k) # default last dim
            cur_generated_ids[cur_len] = torch.LongTensor(new_generated_ids).to(self.device)

            cur_len += 1

            if cur_len == final_decode_len:
                break

        return cur_generated_ids


    def save_checkpoint(self, save_checkpoint_dir, print_log):
        if not os.path.exists(f'{save_checkpoint_dir}'):
            os.makedirs(f'{save_checkpoint_dir}')

        if print_log:
            self.logger.info(f'Start saving checkpoint.')

        torch.save({
            'step': self.global_step,
            'config': self.config,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, f'{save_checkpoint_dir}/checkpoint_{self.global_step}.pt')

        if print_log:
            self.logger.info(f'End saving checkpoint.')


    def save_record(self, tag_value_dict, step):
        for tag, value in tag_value_dict.items():
            self.writer.add_scalar(
                tag,
                value,
                global_step=step)

    def print_current_status(self, key_values):
        self.logger.info(f'cur_step: {self.global_step}')
        self.logger.info(f'cur_lr: {self.config.lr}')
        self.logger.info(f'cur_masking_weight: {self.model.masking_weight}')
        self.logger.info(f'cur_masking_ratio: {self.model.masking_ratio}')
        for key, value in key_values.items():
            self.logger.info(f'{key}: {value}')


    def create_and_register_hook(self):
        self.attn_hooks = []
        for layer_ind, layer in enumerate(self.model.decoder.decoder_layer.layers):
            hook = AttnHook()
            hook.register(layer.multihead_attn)
            self.attn_hooks.append(hook)

def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x