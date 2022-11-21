import argparse
import math
import os
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm

from sum_dist.utils.data.cnndm import DatasetCNNDM
from sum_dist.utils.parse import str2bool
from sum_dist.utils.tokenizer import Tokenizer


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

    def make_spans(self, tokens, window_size):
        spans = []
        for token_ind in range(len(tokens)-window_size+1):
            spans.append((token_ind, token_ind+window_size))
        return spans


def generate_gold_spans(args):
    # set dataset
    dataset = DatasetCNNDM(
        dataset_pkl_path=None, 
        ann_pkl_path=args.cnn_ann_pkl_dir, 
        split='test',
        num_data='')
    m_tokenizer = Tokenizer()
    freq_tool = SpanFreqTool()
    
    # read gold summaries
    gold_spans = [] # DataSize x SeqLen x WindowSize
    for data_ind in tqdm(range(len(dataset))):
        cur_gold_summary = dataset[data_ind]['summary'].replace('\n', ' ').lower()
        cur_gold_tokens = m_tokenizer.tokenize(cur_gold_summary, 'bert', do_stemming=False)
        cur_gold_spans = freq_tool.make_spans(cur_gold_tokens, args.window_size)
        gold_spans.append([' '.join(cur_gold_tokens[span_range[0]:span_range[1]].copy()) for span_range in cur_gold_spans.copy()])

    gold_embeds = []
    for cur_gold_spans in tqdm(gold_spans):
        gold_embeds.append(freq_tool.get_embeds(cur_gold_spans)) # goldSeqLen x E

    assert len(gold_embeds) == len(gold_spans)
    print(len(gold_embeds))

    # record pickle result
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    with open(f'{args.output_dir}/cnndm_gold_spans_embeds.pkl', 'wb') as f:
        pickle.dump({
            'weights': gold_embeds,
            'spans': gold_spans,
        }, f)

    return


def evaluate(args):
    # read attn values & spans
    with open(args.attn_value_file, 'rb') as f:
        a = pickle.load(f)
        values = a['weights'][0] # list of D numpy arrays, D = dataset size
        

        spans = a['tokens'] # list of D lists of spans, D = dataset size

    assert len(spans) == len(values)
    print(len(spans))
    print(len(values))

    # sentence-bert model
    freq_tool = SpanFreqTool()

    # read gold spans/embeds
    gold_spans = []
    if args.freq_in_gold:
        with open(args.gold_pkl_file, 'rb') as f:
            a = pickle.load(f)

            gold_spans = a['spans']
            gold_embeds = a['weights']

        assert len(gold_spans) == len(spans) == len(gold_embeds)

    # to record
    updated_values = []
    updated_spans = []
    freq_idx = []
    attn_idx = []
    prf = []

    # get freq of each article
    for i in tqdm(range(len(spans))):
        cur_spans = spans[i] # S
        cur_values = values[i] # S

        # not count in spans w/ [PAD]
        valid_index = len(cur_spans)
        for idx, sp in enumerate(cur_spans):
            if '[SEP]' in sp:
                valid_index = idx
                break
        cur_spans = cur_spans[1:valid_index]
        cur_values = cur_values[1:valid_index]

        updated_spans.append(cur_spans.copy())
        updated_values.append(cur_values.copy())

        # make embeds
        cur_spans = [' '.join(token) for token in cur_spans]
        cur_embeds = freq_tool.get_embeds(cur_spans) # S x E
        
        assert len(cur_spans) == cur_values.shape[0] == cur_embeds.shape[0]

        num_span = len(cur_spans)
        
        # get indices of top attn spans
        sorted_attn_idx = np.argsort(-cur_values)
        attn_idx.append(sorted_attn_idx.copy())

        # get indices of top sbert freq spans
        cosine_scores = util.pytorch_cos_sim(cur_embeds, cur_embeds) # S x S
        freq = cosine_scores > args.similarity_threshold
        freq_list = torch.sum(freq, dim=0)
        freq_list = freq_list - 1 # S
        freq_list = freq_list.detach().cpu().numpy()
        sorted_freq_idx = np.argsort(-freq_list)
        freq_idx.append(sorted_freq_idx.copy())
        assert num_span == sorted_freq_idx.shape[0]

        # get indices in gold
        if args.freq_in_gold:
            cur_gold_spans = gold_spans[i] # S
            cur_gold_embeds = gold_embeds[i] # goldSeqLen x E

            gold_cosine_scores = util.pytorch_cos_sim(cur_embeds, cur_gold_embeds) # S x goldSeqLen
            count = gold_cosine_scores > args.similarity_threshold
            count = torch.sum(count, dim=1) # S
            count = count >= 1
            in_gold_idx = torch.nonzero(count).flatten()
            in_gold_idx = in_gold_idx.detach().cpu().numpy()

        # get top n of above 2
        if args.m > 1:
            top_freq_num = int(args.m)
        else:
            top_freq_num = math.ceil(args.m * num_span)
        if args.freq_in_gold:
            sorted_freq_idx = [idx for idx in sorted_freq_idx if idx in in_gold_idx]
            if top_freq_num > len(sorted_freq_idx): top_freq_num = len(sorted_freq_idx)
        top_freq_idx = np.array(sorted_freq_idx[:top_freq_num].copy()) # ref

        # how to decide n
        if args.n_based_on_m:
            top_attn_num = top_freq_num
        else:
            if args.n > 1:
                top_attn_num = int(args.n)
            else:
                top_attn_num = math.ceil(args.n * num_span)
        top_attn_idx = sorted_attn_idx[:top_attn_num] # pred

        # get prf
        if len(top_attn_idx) == 0:
            p = 0
        else:
            p = 0
            tensor_top_freq_idx = torch.from_numpy(top_freq_idx).type(torch.LongTensor).to(cosine_scores.device)
            for idx in top_attn_idx:
                tmp_cos_score = cosine_scores[idx]
                tmp_cos_score = torch.index_select(tmp_cos_score, 0, tensor_top_freq_idx)
                is_exist = tmp_cos_score > args.similarity_threshold
                if torch.any(is_exist): p += 1
            p /= len(top_attn_idx)

        if len(top_freq_idx) == 0:
            r = 0
        else:
            r = 0
            tensor_top_attn_idx = torch.from_numpy(top_attn_idx).to(cosine_scores.device)
            for idx in top_freq_idx:
                tmp_cos_score = cosine_scores[idx]
                tmp_cos_score = torch.index_select(tmp_cos_score, 0, tensor_top_attn_idx)
                is_exist = tmp_cos_score > args.similarity_threshold
                if torch.any(is_exist): r += 1
            r /= len(top_freq_idx)

        if (p+r) == 0:
            f = 0
        else:
            f = 2*p*r/(p+r)

        prf.append({
            'p': p,
            'r': r,
            'f': f,
        })

    # get total prf
    total_prf = {
        'p': sum([score['p'] for score in prf])/len(prf),
        'r': sum([score['r'] for score in prf])/len(prf),
        'f': sum([score['f'] for score in prf])/len(prf),
    }
    print(total_prf)

    if args.freq_in_gold:
        assert len(updated_spans) == len(updated_values) == len(freq_idx) == len(attn_idx) == len(prf) == len(gold_spans)
    else:
        assert len(updated_spans) == len(updated_values) == len(freq_idx) == len(attn_idx) == len(prf)

    # record pickle result
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    if args.n_based_on_m:
        out_pkl_name = f'{args.output_dir}/attn_freq_spans_result-freq_in_gold_{args.freq_in_gold}-m{args.m}-n_based_on_m-st{args.similarity_threshold}.pkl'
    else:
        out_pkl_name = f'{args.output_dir}/attn_freq_spans_result-freq_in_gold_{args.freq_in_gold}-m{args.m}-n{args.n}-st{args.similarity_threshold}.pkl'

    with open(out_pkl_name, 'wb') as f:
        pickle.dump({
            'weights': updated_values,
            'spans': updated_spans,
            'freq_idx': freq_idx,
            'attn_idx': attn_idx,
            'prf': prf,
            'total_prf': total_prf,
            'summary_spans': gold_spans,
        }, f)

    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, nargs='?', default='evaluate', choices=['evaluate', 'generate'])
    parser.add_argument('-output_dir', type=str, nargs='?', default='./sum_dist/test_output')

    # generate
    parser.add_argument('-cnn_ann_pkl_dir', type=str, nargs='?', default='./sum_dist/data/preprocess/cnndm-bert-ann.pkl')
    parser.add_argument('-window_size', type=int, nargs='?', default=5) # ws

    # evaluate
    parser.add_argument('-attn_value_file', type=str, nargs='?', default='./sum_dist/test_output/attn_weight-decode1-test.pkl')
    parser.add_argument('-gold_pkl_file', type=str, nargs='?', default='./sum_dist/test_output/cnndm_gold_spans_embeds.pkl')
    parser.add_argument('-freq_in_gold', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('-m', type=float, nargs='?', default=0.05, help='denominator freq top m') # 200, 0.05, 0.1, 0.2, 0.3
    parser.add_argument('-n', type=float, nargs='?', default=200, help='attn top n or top n%') # len x m%
    parser.add_argument('-n_based_on_m', type=str2bool, nargs='?', const=True, default=False, help='if true, args.n is ignored.')
    parser.add_argument('-similarity_threshold', type=float, nargs='?', default=0.5) # threshold

    args = parser.parse_args()


    if args.mode == 'evaluate':
        evaluate(args)
    elif args.mode == 'generate':
        generate_gold_spans(args)
    
    return


if __name__ == '__main__':
    main()
