"""
Copy from `baseline_ext_freq_spans.py`.
Should do the function of `statistics_exp.py`:

Observe in the given training set, if high frequency span in source documents match the spans in gold summaries.
"""

from tqdm import tqdm
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

from sum_dist.utils.tokenizer import Tokenizer
from sum_dist.utils.evaluate import RougeCalculator
from sum_dist.utils.data.cnndm import DatasetCNNDM
from sum_dist.utils.data.xsum import DatasetXSUM

import argparse
import math
import os
import pickle

class SpanFreqTool(object):
    def __init__(self):
        self.sbert = SentenceTransformer('paraphrase-distilroberta-base-v2')
        return

    def make_spans(self, tokens, window_size):
        spans = []
        for token_ind in range(len(tokens)-window_size+1):
            spans.append((token_ind, token_ind+window_size))
        return spans

    def get_embeds(self, sents):
        try:
            embeds = self.sbert.encode(sents, convert_to_tensor=True)
        except:
            return None
        return embeds

    def get_freq(self, tokenized_article, spans, source_window_size, similarity_method, overlap_threshold=None, similarity_threshold=None):
        span_embeds = None
        if similarity_method == 'token_overlap':
            freq_list = [0 for _ in spans]
            for index_a, span_a in enumerate(spans):
                count = 0
                cur_span_tokens = tokenized_article[span_a[0]:span_a[1]]
                for index_b in range(index_a+1, len(spans)):
                    span_b = spans[index_b]
                    cur_source_spans = tokenized_article[span_b[0]:span_b[1]]

                    if self._similarity_token_overlap(cur_span_tokens, cur_source_spans, overlap_threshold):
                        count += 1
                        freq_list[index_a] += 1
                        freq_list[index_b] += 1

        elif similarity_method == 'sbert':
            span_tokens = [' '.join(tokenized_article[span[0]:span[1]]) for span in spans]
            span_embeds = self.get_embeds(span_tokens)
            if span_embeds is None:
                return None, None
            cosine_scores = util.pytorch_cos_sim(span_embeds, span_embeds) # N x N
            freq = cosine_scores > similarity_threshold
            freq_list = torch.sum(freq, dim=0)
            freq_list = freq_list - 1
            freq_list = freq_list.tolist()

        return freq_list, span_embeds

    def get_high_freq_spans(self, freq_list, spans, threshold_percent, embeds=None):
        assert len(freq_list) == len(spans)
        high_freq_indexes = sorted(range(len(freq_list)), key=lambda i: freq_list[i], reverse=True)[:]
        threshold_index = math.ceil(len(freq_list)*threshold_percent)
        high_freq_indexes = high_freq_indexes[:threshold_index]
        high_freq_spans = [spans[index] for index in high_freq_indexes]

        high_freq_embeds = None
        if embeds is not None:
            high_freq_embeds = [embeds[index] for index in high_freq_indexes]
            high_freq_embeds = torch.stack(high_freq_embeds, dim=0)

        return high_freq_spans, high_freq_embeds

    def get_A_in_B(self, spans_A, spans_B, tokenized_A, tokenized_B, similarity_method, embeds_A=None, embeds_B=None, overlap_threshold=None, similarity_threshold=None):
        """
        A: summary spans
        B: source spans
        calculate if a span in A appear in B based on the number of overlapping tokens in each span.
        """
        if embeds_A is not None:
            assert len(spans_A) == len(embeds_A)
        if embeds_B is not None:
            assert len(spans_B) == len(embeds_B)

        count=0
        if similarity_method == 'token_overlap':
            for index_A, summary_span in enumerate(spans_A):
                cur_summary_span_tokens = tokenized_A[summary_span[0]:summary_span[1]]
                for index_B, source_span in enumerate(spans_B):
                    cur_source_span_tokens = tokenized_B[source_span[0]:source_span[1]]
                    if self._similarity_token_overlap(cur_summary_span_tokens, cur_source_span_tokens, overlap_threshold):
                        count += 1
                        break
        elif similarity_method == 'sbert':
            cosine_scores = util.pytorch_cos_sim(embeds_A, embeds_B) # |A| x |B|
            count = cosine_scores > similarity_threshold
            count = torch.sum(count, dim=1)
            count = count >= 1
            count = torch.sum(count).item()

        return count

    def _similarity_token_overlap(self, span_A, span_B, overlap_threshold):
        """
        Args:
            - `span_A` (List[str])
            - `span_B` (List[str])
            - `overlap_threshold` (float)
        """
        count = 0
        for token in span_A:
            if token in span_B: count += 1
            if count >= overlap_threshold:
                return True
            else:
                return False

    def _similarity_sbert(self, embeds_A, embeds_B, similarity_threshold):
        """
        Args:
            - `span_A` (List[str])
            - `span_B` (List[str])
            - `overlap_threshold` (float)
        """
        cosine_scores = util.pytorch_cos_sim(embeds_A, embeds_B)
        if cosine_scores >= similarity_threshold:
            return True
        else:
            return False

    def combine_spans(self, article_tokens, spans, max_len):
        used_token_inds = [0 for _ in range(len(article_tokens))]
        spans = sorted(spans)
        for span in spans:
            used_token_inds[span[0]:span[1]] = [1 for _ in range(span[1]-span[0])]
            if sum(used_token_inds) >= max_len:
                break

        summary_tokens = []
        for ind, if_used in enumerate(used_token_inds):
            if if_used == 1:
                summary_tokens.append(article_tokens[ind])
        return summary_tokens

    def combine_sents(self, article_tokens, article_spans, span_freq, sent_spans, max_len):
        token_scores = [0 for _ in range(len(article_tokens))]
        for span_id, span in enumerate(article_spans):
            cur_span_freq = span_freq[span_id]
            token_scores[span[0]:span[1]] += [cur_span_freq for _ in range(span[1]-span[0])]

        sent_scores = []
        for sent_ind, sent_span in enumerate(sent_spans):
            if sent_span[1]-sent_span[0] > 0:
                cur_sent_score = sum(token_scores[sent_span[0]:sent_span[1]])/(sent_span[1]-sent_span[0])
            else:
                cur_sent_score = 0
            sent_scores.append(cur_sent_score)

        summary_tokens = []
        high_score_sent_indexes = np.argsort(sent_scores).tolist() # low to high
        high_score_sent_indexes.reverse() # low to high
        selected_sent_indexes = []
        for sent_ind in high_score_sent_indexes:
            cur_sent = article_tokens[sent_spans[sent_ind][0]:sent_spans[sent_ind][1]]
            summary_tokens += cur_sent
            selected_sent_indexes.append(sent_ind)
            if len(summary_tokens) >= max_len: break

        selected_sent_indexes = sorted(selected_sent_indexes)

        summary_tokens = []
        for sent_ind in selected_sent_indexes:
            cur_sent = article_tokens[sent_spans[sent_ind][0]:sent_spans[sent_ind][1]]
            summary_tokens += cur_sent
            if len(summary_tokens) >= max_len: break

        return summary_tokens


def run_baseline_high(args):
    m_tokenizer = Tokenizer()
    span_freq_tool = SpanFreqTool()

    # set dataset
    if args.dataset == 'cnndm':
        dataset = DatasetCNNDM(
            dataset_pkl_path=None, 
            ann_pkl_path=args.cnn_ann_pkl_dir, 
            split=args.split,
            num_data='')
    elif args.dataset == 'xsum':
        dataset = DatasetXSUM(
            dataset_pkl_path=None, 
            ann_pkl_path=args.xsum_ann_pkl_dir,
            split=args.split,
            num_data=''
        )
    
    do_stemming = True
    if args.similarity_method == 'sbert':
        do_stemming = False
    
    final_summaries = []
    gold_summaries = []
    for data_ind in tqdm(range(len(dataset))):
        cur_gold_summary = dataset[data_ind]['summary'].replace('\n', ' ').lower()
        cur_source = dataset[data_ind]['article']

        tokens = m_tokenizer.tokenize(cur_source, args.tokenize_method, do_stemming=do_stemming)
        spans = span_freq_tool.make_spans(tokens, args.window_size)
        freq_list, embeds = span_freq_tool.get_freq(tokens, spans, args.window_size, args.similarity_method, args.overlap_threshold, args.similarity_threshold)
        if freq_list is None and embeds is None: continue
        assert len(spans) == len(freq_list)

        sent_spans = []
        start_idx = 0
        for sent in m_tokenizer.sent_tokenize(cur_source):
            cur_sent_tokens = m_tokenizer.tokenize(sent, args.tokenize_method, do_stemming=do_stemming)
            sent_spans.append((start_idx, start_idx+len(cur_sent_tokens)))
            start_idx += len(cur_sent_tokens)

        summary_tokens = span_freq_tool.combine_sents(tokens, spans, freq_list, sent_spans, args.output_token_num)
        summary = m_tokenizer.tokens_to_string(summary_tokens, args.tokenize_method)

        final_summaries.append(summary)
        gold_summaries.append(cur_gold_summary)

    assert len(final_summaries) == len(gold_summaries)

    # setting output dir/filename
    cur_output_dir = f'{args.output_dir}/baseline/high/{args.dataset}/{args.tokenize_method}/{args.similarity_method}'
    if not os.path.exists(cur_output_dir):
        os.makedirs(cur_output_dir)

    threshold_str = f'{args.overlap_threshold}' if args.similarity_method == 'token_overlap' else f'{args.similarity_threshold}'
    settings_str = f'ws{args.window_size}-len{args.output_token_num}-threshold{threshold_str}'

    # write baseline summaries
    with open(f'{cur_output_dir}/prediction-{settings_str}.txt', 'w') as f:
        for line_ind, summary in enumerate(final_summaries):
            f.write(summary)
            if line_ind != len(final_summaries) - 1:
                f.write('\n')

    # write gold summaries
    with open(f'{cur_output_dir}/gold-{settings_str}.txt', 'w') as f:
        for line_ind, summary in enumerate(gold_summaries):
            f.write(summary)
            if line_ind != len(gold_summaries) - 1:
                f.write('\n')

    # set rouge calculator
    prediction_dir = f'{cur_output_dir}/{settings_str}/prediction'
    gold_dir = f'{cur_output_dir}/{settings_str}/gold'
    rouge_calculator = RougeCalculator(
        prediction_dir=prediction_dir,
        gold_dir=gold_dir,
        prediction_prefix='prediction', 
        gold_prefix='gold')


    # write target
    assert len(final_summaries) == len(gold_summaries)
    rouge_calculator.convert_article_to_rouge_file(
        gold_summaries, 
        is_prediction=False,
        gold_dir=gold_dir)

    # write prediction
    rouge_calculator.convert_article_to_rouge_file(
        final_summaries, 
        is_prediction=True, 
        prediction_dir=prediction_dir)

    # calculate ROUGE
    scores = rouge_calculator.get_score(
        prediction_dir=prediction_dir,
        gold_dir=gold_dir)

    print(scores)

    # write score
    with open(f'{cur_output_dir}/score-{settings_str}.txt', 'w') as f:
        f.write(str(scores))
        f.write('\n')

    return


def run_statistics(args):
    m_tokenizer = Tokenizer()
    span_freq_tool = SpanFreqTool()

    # set dataset
    if args.dataset == 'cnndm':
        dataset = DatasetCNNDM(
            dataset_pkl_path=None, 
            ann_pkl_path=args.cnn_ann_pkl_dir, 
            split=args.split,
            num_data='')
    elif args.dataset == 'xsum':
        dataset = DatasetXSUM(
            dataset_pkl_path=None, 
            split=args.split,
            num_data=''
        )

    do_stemming = True
    if args.similarity_method == 'sbert':
        do_stemming = False

    summary_in_source_ratio = []
    source_in_summary_ratio = []

    for data_ind in tqdm(range(len(dataset))):
        cur_source = dataset[data_ind]['article']
        cur_summary = dataset[data_ind]['summary']

        # get source-span frequency
        source_tokens = m_tokenizer.tokenize(cur_source, args.tokenize_method, do_stemming=do_stemming)
        source_spans = span_freq_tool.make_spans(source_tokens, args.window_size)
        source_freq_list, source_embeds = span_freq_tool.get_freq(
            tokenized_article=source_tokens, 
            spans=source_spans, 
            source_window_size=args.window_size, 
            similarity_method=args.similarity_method, 
            overlap_threshold=args.overlap_threshold, 
            similarity_threshold=args.similarity_threshold)
        if source_freq_list is None and source_embeds is None: continue
        source_high_freq_spans, source_high_freq_embeds = span_freq_tool.get_high_freq_spans(source_freq_list, source_spans, args.high_freq_percent, source_embeds)

        # get summary-span
        summary_tokens = m_tokenizer.tokenize(cur_summary, args.tokenize_method, do_stemming=do_stemming)
        summary_spans = span_freq_tool.make_spans(summary_tokens, args.window_size)
        summary_embeds = span_freq_tool.get_embeds([' '.join(summary_tokens[span[0]:span[1]]) for span in summary_spans])
        if summary_embeds is None: continue

        # compare source spans & summary spans
        num_summary_span = len(summary_spans)
        num_high_freq_source_span = len(source_high_freq_spans)
        num_summary_span_appear_in_source = span_freq_tool.get_A_in_B(
            spans_A=summary_spans,
            spans_B=source_high_freq_spans,
            tokenized_A=summary_tokens,
            tokenized_B=source_tokens,
            embeds_A=summary_embeds,
            embeds_B=source_high_freq_embeds,
            similarity_method=args.similarity_method,
            overlap_threshold=args.overlap_threshold,
            similarity_threshold=args.similarity_threshold,
        )
        num_source_span_appear_in_summary = span_freq_tool.get_A_in_B(
            spans_A=source_high_freq_spans,
            spans_B=summary_spans,
            tokenized_A=source_tokens,
            tokenized_B=summary_tokens,
            embeds_A=source_high_freq_embeds,
            embeds_B=summary_embeds,
            similarity_method=args.similarity_method,
            overlap_threshold=args.overlap_threshold,
            similarity_threshold=args.similarity_threshold,
        )

        assert num_summary_span_appear_in_source <= num_summary_span

        if num_summary_span != 0:
            summary_in_source_ratio.append(num_summary_span_appear_in_source/num_summary_span)
        if num_high_freq_source_span != 0:
            source_in_summary_ratio.append(num_source_span_appear_in_summary/num_high_freq_source_span)

    print(f'data num-sum in source: {len(summary_in_source_ratio)}')
    print(f'data num-source in sum: {len(source_in_summary_ratio)}')

    # setting output dir/filename
    cur_output_dir = f'{args.output_dir}/statistics'
    if not os.path.exists(cur_output_dir):
        os.makedirs(cur_output_dir)

    threshold_str = f'{args.overlap_threshold}' if args.similarity_method == 'token_overlap' else f'{args.similarity_threshold}'
    settings_str = f'{args.dataset}-t_{args.tokenize_method}-s{args.similarity_method}-ws{args.window_size}-threshold{threshold_str}-hfp{args.high_freq_percent}'

    # plot
    plt.figure()
    plt.hist(summary_in_source_ratio, bins=20)
    plt.xlim(0,1)
    plt.savefig(f'{cur_output_dir}/freq_span_ratio_hist-sum_in_source-{settings_str}.png')

    plt.figure()
    plt.hist(source_in_summary_ratio, bins=20)
    plt.xlim(0,1)
    plt.savefig(f'{cur_output_dir}/freq_span_ratio_hist-source_in_sum-{settings_str}.png')

    # write result
    with open(f'{cur_output_dir}/freq_span_ratio-{settings_str}.pkl', 'wb') as f:
        pickle.dump({
            'sum_in_source': summary_in_source_ratio,
            'source_in_sum': source_in_summary_ratio}, f)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, nargs='?', default='baseline_high', choices=['baseline_high', 'statistics'])

    # run statistics
    parser.add_argument('-dataset', type=str, nargs='?', default='cnndm', choices=['cnndm', 'xsum'])
    parser.add_argument('-split', type=str, nargs='?', default='test', choices=['train', 'test', 'val', 'validation']) # only run_statistics() should use `train`, otherwise `test`
    parser.add_argument('-cnn_ann_pkl_dir', type=str, nargs='?', default='./sum_dist/data/preprocess/cnndm-bert-ann.pkl')
    parser.add_argument('-xsum_ann_pkl_dir', type=str, nargs='?', default='./sum_dist/data/preprocess/xsum-bert-ann.pkl')
    parser.add_argument('-output_dir', type=str, nargs='?', default='./sum_dist/output')
    parser.add_argument('-tokenize_method', type=str, nargs='?', default='nltk', choices=['bert', 'space', 'nltk']) # t
    parser.add_argument('-similarity_method', type=str, nargs='?', default='sbert', choices=['sbert', 'token_overlap']) # s
    parser.add_argument('-window_size', type=int, nargs='?', default=5) # ws
    parser.add_argument('-overlap_threshold', type=int, nargs='?', default=2) # threshold
    parser.add_argument('-similarity_threshold', type=float, nargs='?', default=0.5) # threshold
    parser.add_argument('-high_freq_percent', type=float, nargs='?', default=0.03) # hfp

    # run baseline
    parser.add_argument('-output_token_num', type=int, nargs='?', default=50) # len
    args = parser.parse_args()

    if args.mode == 'baseline_high':
        print(f'dataset: {args.dataset}')
        print(f'tokenize: {args.tokenize_method}')
        print(f'similarity method: {args.similarity_method}')
        print('-----------------------------------')
        print(f'window size: {args.window_size}')
        print(f'similarity threshold: {args.similarity_threshold}')
        print(f'overlap threshold: {args.overlap_threshold}')
        print(f'high_freq_percent: {args.high_freq_percent}')
        print(f'output token len: {args.output_token_num}')
        run_baseline_high(args)

    elif args.mode == 'statistics':
        print(f'dataset: {args.dataset}')
        print(f'tokenize: {args.tokenize_method}')
        print(f'similarity method: {args.similarity_method}')
        print('-----------------------------------')
        print(f'window size: {args.window_size}')
        print(f'similarity threshold: {args.similarity_threshold}')
        print(f'overlap threshold: {args.overlap_threshold}')
        print(f'high_freq_percent: {args.high_freq_percent}')
        run_statistics(args)

    return

if __name__ == '__main__':
    main()
