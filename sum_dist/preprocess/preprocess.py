"""
Specify a tokenizer & a dataset.
Do sentence segmentation (Spacy) & tokenization on all articles in the dataset.
Output preprocessed pickle files containing sentence segmentation indexes & tokenized indexes.
"""

import spacy
from tqdm import tqdm
from transformers import BertTokenizerFast, AutoTokenizer
from datasets import load_dataset

import argparse
from itertools import repeat
from multiprocessing import Pool
import json
import os
import pickle
import re
import sys
import time

from sum_dist.configs import MConfigs


def clean_cnndm_multi(split_key, dataset, index, tokenizer, sent_tokenizer):
    cleaned_dataset = {}

    data = dataset[split_key][index]

    # remove `... (CNN) --`
    article = data['article']
    article_start_idx = 0
    
    target_re = '\([^(\)).]+\) -- '
    match = re.search(target_re, article)
    if match != None:
        article_start_idx = match.end()
    if match is None or article_start_idx > 50:
        article_start_idx = 0
        target_str = '(CNN)'
        remove_ind = article.find(target_str)
        article_start_idx = remove_ind + len(target_str)

    if article_start_idx > 50:
        article_start_idx = 0

    if article_start_idx > 0:
        article = article[article_start_idx:]

    # sentence tokenizer
    sent_spans = sent_tokenizer(article)

    cleaned_dataset[data['id']] = {
        'article_start_idx': article_start_idx,
        'sentence_spans': [
            (sent_span.start_char, sent_span.end_char) for sent_span in sent_spans.sents
        ],
        'token_lens': [
            len(tokenizer(
                text=sent_span.text, 
                add_special_tokens=False, 
                return_token_type_ids=False, 
                return_attention_mask=False)['input_ids']) 
            for sent_span in sent_spans.sents],
    }

    return cleaned_dataset


def clean_xsum_multi(split_key, dataset, index, tokenizer, sent_tokenizer):
    cleaned_dataset = {}

    data = dataset[split_key][index]

    # remove `... (CNN) --`
    article = data['document']
    article = article.replace('[', '')
    article = article.replace(']', '')
    article = article.replace('\n', ' ')

    article_start_idx = 0

    # sentence tokenizer
    sent_spans = sent_tokenizer(article)

    cleaned_dataset[data['id']] = {
        'article_start_idx': article_start_idx,
        'sentence_spans': [
            (sent_span.start_char, sent_span.end_char) for sent_span in sent_spans.sents
        ],
        'token_lens': [
            len(tokenizer(
                text=sent_span.text, 
                add_special_tokens=False, 
                return_token_type_ids=False, 
                return_attention_mask=False)['input_ids']) 
            for sent_span in sent_spans.sents],
    }

    return cleaned_dataset


def clean_mlsum_es_multi(split_key, dataset, index, tokenizer, sent_tokenizer):
    cleaned_dataset = {}

    data = dataset[split_key][index]

    article = data['text']
    article = article.replace('\n', ' ')

    article_start_idx = 0

    # sentence tokenizer
    sent_spans = sent_tokenizer(article)

    cleaned_dataset[data['url']] = {
        'article_start_idx': article_start_idx,
        'sentence_spans': [
            (sent_span.start_char, sent_span.end_char) for sent_span in sent_spans.sents
        ],
        'token_lens': [
            len(tokenizer(
                text=sent_span.text, 
                add_special_tokens=False, 
                return_token_type_ids=None, 
                return_attention_mask=False)['input_ids']) 
            for sent_span in sent_spans.sents],
    }

    return cleaned_dataset

    
def clean_mlsum_de_multi(split_key, dataset, index, tokenizer, sent_tokenizer):
    cleaned_dataset = {}

    data = dataset[split_key][index]

    article = data['text']
    article = article.replace('\n', ' ')

    article_start_idx = 0

    # sentence tokenizer
    sent_spans = sent_tokenizer(article)

    cleaned_dataset[data['url']] = {
        'article_start_idx': article_start_idx,
        'sentence_spans': [
            (sent_span.start_char, sent_span.end_char) for sent_span in sent_spans.sents
        ],
        'token_lens': [
            len(tokenizer(
                text=sent_span.text, 
                add_special_tokens=False, 
                return_token_type_ids=None, 
                return_attention_mask=False)['input_ids']) 
            for sent_span in sent_spans.sents],
    }

    return cleaned_dataset


def clean_mlsum_ru_multi(split_key, dataset, index, tokenizer, sent_tokenizer):
    cleaned_dataset = {}

    data = dataset[split_key][index]

    article = data['text']
    article = article.replace('\n', ' ')

    article_start_idx = 0

    # sentence tokenizer
    sent_spans = sent_tokenizer(article)

    cleaned_dataset[data['url']] = {
        'article_start_idx': article_start_idx,
        'sentence_spans': [
            (sent_span.start_char, sent_span.end_char) for sent_span in sent_spans.sents
        ],
        'token_lens': [
            len(tokenizer(
                text=sent_span.text, 
                add_special_tokens=False, 
                return_token_type_ids=None, 
                return_attention_mask=False)['input_ids']) 
            for sent_span in sent_spans.sents],
    }

    return cleaned_dataset


def clean_wiki_en_multi(split_key, dataset, index, tokenizer, sent_tokenizer):
    cleaned_dataset = {}

    data = dataset[split_key][index]

    article = data['text']
    article = article.replace('\n', ' ')

    article_start_idx = 0

    # sentence tokenizer
    sent_spans = sent_tokenizer(article)

    cleaned_dataset[data['title']] = {
        'article_start_idx': article_start_idx,
        'sentence_spans': [
            (sent_span.start_char, sent_span.end_char) for sent_span in sent_spans.sents
        ],
        'token_lens': [
            len(tokenizer(
                text=sent_span.text, 
                add_special_tokens=False, 
                return_token_type_ids=None, 
                return_attention_mask=False)['input_ids']) 
            for sent_span in sent_spans.sents],
    }

    return cleaned_dataset


def clean_arxiv_multi(split_key, dataset, index, tokenizer):
    cleaned_dataset = {}
    data = dataset[split_key][index]
    article = data['article_text']
    article_start_idx = 0
    cleaned_article = []
    sent_spans = []
    cur_start_char = 0

    for line in article:
        cleaned_article.append(line)
        sent_spans.append((cur_start_char, cur_start_char+len(line)))
        cur_start_char += (len(line)+1)

    cleaned_dataset[data['article_id']] = {
        'article_start_idx': article_start_idx,
        'sentence_spans': sent_spans,
        'token_lens': [
            len(tokenizer(
                text=line, 
                add_special_tokens=False, 
                return_token_type_ids=None, 
                return_attention_mask=False)['input_ids']) 
            for line in cleaned_article],
    }

    return cleaned_dataset


def main_multi():
    # read args & configs
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-dataset', 
        type=str, 
        nargs='?', 
        default='cnndm', 
        choices=['cnndm', 'xsum', 'mlsum_es', 'mlsum_de', 'mlsum_ru', 'wiki_en', 'arxiv'], 
        help='select a dataset to preprocess')
    parser.add_argument(
        '-read_config', 
        type=str, 
        nargs='?', 
        default='./sum_dist/config_preliminary.json', 
        help='specify config json file')
    parser.add_argument(
        '-tokenizer', 
        type=str, 
        nargs='?', 
        default='bert', 
        choices=['bert'], 
        help='select a tokenizer')
    parser.add_argument(
        '-save_dir', 
        type=str, 
        nargs='?', 
        default='./sum_dist/data/preprocess', 
        help='specify preprocessed file path')
    parser.add_argument(
        '-num_worker', 
        type=int, 
        nargs='?', 
        default=10)
    args = parser.parse_args()

    configs = MConfigs().load_json(load_dir=args.read_config)

    # check write folder path
    if not os.path.exists(f'{args.save_dir}'):
        os.makedirs(f'{args.save_dir}')
    else:
        file_names = os.listdir(f'{args.save_dir}')
        if f'{args.dataset}-{args.tokenizer}-ann.pkl' in file_names:
            raise RuntimeWarning(
                f'File inside the folder may be overwritten: {args.save_dir}/{args.dataset}-{args.tokenizer}-ann.pkl')

    # read dataset
    if args.dataset == 'cnndm':
        dataset = load_dataset('cnn_dailymail', '3.0.0')
        tokenizer = BertTokenizerFast.from_pretrained(configs.bert_version)
        split_work_func = clean_cnndm_multi
    elif args.dataset == 'xsum':
        dataset = load_dataset(args.dataset)
        tokenizer = BertTokenizerFast.from_pretrained(configs.bert_version)
        split_work_func = clean_xsum_multi
    elif args.dataset == 'mlsum_es':
        dataset = load_dataset('mlsum', 'es')
        tokenizer = AutoTokenizer.from_pretrained(configs.bert_version)
        split_work_func = clean_mlsum_es_multi
    elif args.dataset == 'mlsum_de':
        dataset = load_dataset('mlsum', 'de')
        tokenizer = AutoTokenizer.from_pretrained(configs.bert_version)
        split_work_func = clean_mlsum_de_multi
    elif args.dataset == 'mlsum_ru':
        dataset = load_dataset('mlsum', 'ru')
        tokenizer = AutoTokenizer.from_pretrained(configs.bert_version)
        split_work_func = clean_mlsum_ru_multi
    elif args.dataset == 'wiki_en':
        dataset = load_dataset('wikipedia', '20200501.en')
        tokenizer = BertTokenizerFast.from_pretrained(configs.bert_version)
        split_work_func = clean_wiki_en_multi
    elif args.dataset == 'arxiv':
        dataset = {}
        with open('./sum_dist/data/arxiv_data/arxiv-dataset/arxiv-dataset/train.txt', 'r') as f:
            dataset['train'] = []
            for ind, line in enumerate(f):
                json_data_train = json.loads(line)
                dataset['train'].append(json_data_train)
        with open('./sum_dist/data/arxiv_data/arxiv-dataset/arxiv-dataset/val.txt', 'r') as f:
            dataset['validation'] = []
            for ind, line in enumerate(f):
                json_data_validation = json.loads(line)
                dataset['validation'].append(json_data_validation)
        with open('./sum_dist/data/arxiv_data/arxiv-dataset/arxiv-dataset/test.txt', 'r') as f:
            dataset['test'] = []
            for ind, line in enumerate(f):
                json_data_test = json.loads(line)
                dataset['test'].append(json_data_test)
        tokenizer = BertTokenizerFast.from_pretrained(configs.bert_version)
        split_work_func = clean_arxiv_multi

    # load sentence tokenizer
    sent_tokenizer = spacy.load(configs.preprocess_spacy_model) # here
    processed = {}

    # split work
    # generate preprocessed pickle file by dataset
    for split_key in dataset.keys():
        processed[split_key] = {}
        start_time = time.time()
        
        with Pool(processes=args.num_worker) as pool:
            if args.dataset == 'arxiv':
                res = pool.starmap(
                    split_work_func, 
                    zip(
                        repeat(split_key), 
                        repeat(dataset),
                        range(len(dataset[split_key])), 
                        repeat(tokenizer), 
                    )
                )
            else:
                res = pool.starmap(
                    split_work_func, 
                    zip(
                        repeat(split_key), 
                        repeat(dataset),
                        range(len(dataset[split_key])), 
                        repeat(tokenizer), 
                        repeat(sent_tokenizer)
                    )
                )

        for res_dict in res:
            processed[split_key].update(res_dict)

        print(split_key)
        print(len(processed[split_key]))
        print(time.time()-start_time)

    # write pickle file
    with open(f'{args.save_dir}/{args.dataset}-{args.tokenizer}-ann.pkl', 'wb') as f:
        pickle.dump(processed, f)
    
    return

if __name__ == '__main__':
    main_multi()
