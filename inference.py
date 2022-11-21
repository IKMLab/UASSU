"""
Load models in given folder. 
Inference the test set using each model.
Save all the decoding results, attention weights(pkl).
"""

import numpy as np
from transformers import BertTokenizer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import argparse
import os
import pickle
import random
import re

from sum_dist.configs import MConfigs
from sum_dist.utils.evaluate import RougeCalculator
from sum_dist.utils.data.collate_fn import InitCollate
from sum_dist.models.encoder import TransformerEncoder
from sum_dist.models.decoder import TransformerDecoder
from sum_dist.models.seq2seq import Seq2seqModel
from sum_dist.trainer import Trainer
from sum_dist.utils.parse import str2bool
from sum_dist.utils.data.cnndm import DatasetCNNDM
from sum_dist.utils.data.xsum import DatasetXSUM
from sum_dist.utils.data.mlsum import DatasetMLSUMde, DatasetMLSUMes, DatasetMLSUMru
import sum_dist.utils.logging as logging


logger = logging.get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()

    # run time settings
    parser.add_argument('-dataset', type=str, nargs='?', default='cnndm', choices=['cnndm', 'xsum', 'mlsum_de', 'mlsum_es', 'mlsum_ru', 'wiki_en', 'arxiv'])
    parser.add_argument('-exp_name', type=str, nargs='?', default='transformer22/lg/window5_s1-mask-my_loss_masking_pos-span_concat')
    parser.add_argument('-run_attn', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('-filter_len', type=int, nargs='?', default=200)
    parser.add_argument('-decoding_max_len', type=int, nargs='?', default=500)
    parser.add_argument('-decoding_target_seq_len', type=int, nargs='?', default=100)
    parser.add_argument('-decoding_times', type=int, nargs='?', default=2, choices=[1, 2])
    parser.add_argument('-num_data_test', nargs='?', default='')
    parser.add_argument('-use_high_rouge', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('-use_high_freq', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('-similarity_threshold', type=float, nargs='?', default=0.5)
    parser.add_argument('-write_source', type=str2bool, nargs='?', const=True, default=True)

    # training settings
    parser.add_argument('-batch_size', type=int, nargs='?', default=1)
    parser.add_argument('-device', type=str, nargs='?', default='cuda:0')

    # training paths
    parser.add_argument('-load_config_dir', type=str, nargs='?', default=None)

    # rouge paths
    parser.add_argument('-prediction_file_prefix', type=str, nargs='?', default='prediction')
    parser.add_argument('-target_file_prefix', type=str, nargs='?', default='gold')

    # dataset paths
    parser.add_argument('-cnn_ann_pkl_dir', type=str, nargs='?', default='./sum_dist/data/preprocess/cnndm-bert-ann.pkl')
    parser.add_argument('-xsum_ann_pkl_dir', type=str, nargs='?', default='./sum_dist/data/preprocess/xsum-bert-ann.pkl')
    parser.add_argument('-mlsum_de_ann_pkl_dir', type=str, nargs='?', default='./sum_dist/data/preprocess/mlsum_de-bert-ann.pkl')
    parser.add_argument('-mlsum_es_ann_pkl_dir', type=str, nargs='?', default='./sum_dist/data/preprocess/mlsum_es-bert-ann.pkl')
    parser.add_argument('-mlsum_ru_ann_pkl_dir', type=str, nargs='?', default='./sum_dist/data/preprocess/mlsum_ru-bert-ann.pkl')
    parser.add_argument('-arxiv_dataset_dir', type=str, nargs='?', default='./sum_dist/data/arxiv_data/arxiv-dataset/arxiv-dataset')
    parser.add_argument('-arxiv_ann_pkl_dir', type=str, nargs='?', default='./sum_dist/data/preprocess/arxiv-bert-ann.pkl')
    parser.add_argument('-wiki_en_ann_pkl_dir', type=str, nargs='?', default='./sum_dist/data/preprocess/wiki_en-bert-ann.pkl')

    # inference (specify if needed)
    parser.add_argument('-decoding_method', type=str, nargs='?', default=None)
    parser.add_argument('-k', type=int, nargs='?', default=None)
    parser.add_argument('-beam_size', type=int, nargs='?', default=None)
    
    args = parser.parse_args()
    logger.info(args)

    device = torch.device(args.device)
    logger.info(f'Using device: {device}')

    """
    Set once.
    """

    # set dataset
    split = 'test'
    num_data = args.num_data_test

    if args.dataset == 'cnndm':
        dataset = DatasetCNNDM(
            dataset_pkl_path=None, 
            ann_pkl_path=args.cnn_ann_pkl_dir, 
            split=split,
            num_data=num_data
        )
    elif args.dataset == 'xsum':
        dataset = DatasetXSUM(
            dataset_pkl_path=None, 
            ann_pkl_path=args.xsum_ann_pkl_dir, 
            split=split,
            num_data=num_data
        )
    elif args.dataset == 'mlsum_de':
        dataset = DatasetMLSUMde(
            dataset_pkl_path=None,
            ann_pkl_path=args.mlsum_de_ann_pkl_dir,
            num_data=num_data,
            split=split)
    elif args.dataset == 'mlsum_es':
        dataset = DatasetMLSUMes(
        dataset_pkl_path=None,
        ann_pkl_path=args.mlsum_es_ann_pkl_dir,
        num_data=num_data,
        split=split)
    elif args.dataset == 'mlsum_ru':
        dataset = DatasetMLSUMru(
        dataset_pkl_path=None,
        ann_pkl_path=args.mlsum_ru_ann_pkl_dir,
        num_data=num_data,
        split=split)

    logger.info('Loading dataset done.')

    # make output dir
    if not os.path.exists(f'./sum_dist/output/inference/{args.exp_name}'):
        os.makedirs(f'./sum_dist/output/inference/{args.exp_name}')

    # write all(unsampled complete) input & target
    if args.write_source:
        split_filename = 'test'
        output_source_filename = f'./sum_dist/output/inference/{args.exp_name}/source_all-{split_filename}.txt'
        output_target_filename = f'./sum_dist/output/inference/{args.exp_name}/target_all-{split_filename}.txt'

        with open(output_source_filename, 'w') as source_f, \
            open(output_target_filename, 'w') as target_f:
            for article_ind in range(len(dataset)):
                instance = dataset[article_ind]

                if len(instance['article']) < 50:
                    continue

                source_f.write(instance['article'][instance['start_idx']:].replace('\n', ' '))
                target_f.write(instance['summary'].strip('\n').replace('\n', ' [NEWLINE] '))
                
                if article_ind < len(dataset) - 1:
                    source_f.write('\n')
                    target_f.write('\n')

    # list all candidate checkpoints
    checkpoint_idx_lst = [int(f[f.find('_')+1:f.find('.')]) for f in os.listdir(f'./sum_dist/checkpoint/{args.exp_name}') if os.path.isfile(os.path.join(f'./sum_dist/checkpoint/{args.exp_name}', f))]
    checkpoint_idx_lst = sorted(checkpoint_idx_lst)

    for checkpoint_idx in tqdm(checkpoint_idx_lst):

        """
        Set every time loading new checkpoint.
        """

        # set rouge calculator
        rouge_calculator = RougeCalculator(
            prediction_dir=f'./sum_dist/output/inference/{args.exp_name}/checkpoint_{checkpoint_idx}/prediction',# args.prediction_dest, 
            gold_dir=f'./sum_dist/output/inference/{args.exp_name}/checkpoint_{checkpoint_idx}/gold', # args.target_dest, 
            prediction_prefix=args.prediction_file_prefix, 
            gold_prefix=args.target_file_prefix)

        logger.info('Setting ROUGE calculator done.')

        config = MConfigs()
        tokenizer = None
        collate_fn = None
        encoder = None
        decoder = None

        # load config
        if f'./sum_dist/checkpoint/{args.exp_name}/checkpoint_{checkpoint_idx}.pt' is not None and os.path.exists(f'./sum_dist/checkpoint/{args.exp_name}/checkpoint_{checkpoint_idx}.pt'):
            
            checkpoint = torch.load(f'./sum_dist/checkpoint/{args.exp_name}/checkpoint_{checkpoint_idx}.pt')
            config = checkpoint['config']
            logger.info(f'Load checkpoint config from: ./sum_dist/checkpoint/{args.exp_name}/checkpoint_{checkpoint_idx}.pt')
        
        if args.load_config_dir is not None and os.path.exists(args.load_config_dir):
            config = config.load_json(load_dir=args.load_config_dir)
            logger.info(f'Load json config from: {args.load_config_dir}')
        
        # TODO: here
        config.update({
            'decoding_target_seq_len': args.decoding_target_seq_len,
            'decoding_times': args.decoding_times,
        })
        if args.decoding_method:
            config.update({"decoding_methods": [args.decoding_method]})
        if args.k:
            config.update({"k": args.k})
        if args.beam_size:
            config.update({"beam_size": args.beam_size})
        if 'seed' not in config.__dict__.keys():
            config.update({'seed': 37})

        logger.info(config.__dict__)

        # set seed
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # set encoder
        tokenizer = BertTokenizer.from_pretrained(config.bert_version)
        collate_fn = InitCollate(
            tokenizer=tokenizer, 
            encoder_max_seq_len=config.seq_len,
            decoder_max_seq_len=args.decoding_max_len,
            target_seq_len=args.decoding_target_seq_len,
            encoder_sampling_len_levels=config.sampling_length_levels,
            cur_encoder_sampling_len_level=config.cur_sampling_length_level,
            inference_mode=True)
        encoder = TransformerEncoder(
            embedding_dim=config.word_embed_size, 
            num_layer=config.encoder_num_layer, 
            num_head=config.encoder_num_head, 
            dim_feedforward=config.encoder_ff_embed_size, 
            decoder_dropout=config.encoder_dropout, 
            activation=config.encoder_activation,
            num_embeddings=len(tokenizer), 
            embeddings=None)

        # set decoder
        decoder = TransformerDecoder(
            encoder_embed_size=config.repr_embed_size, 
            vocab_size=len(tokenizer),
            num_layer=config.decoder_num_layer, 
            num_head=config.decoder_num_head, 
            dim_feedforward=config.decoder_ff_embed_size, 
            decoder_dropout=config.decoder_dropout, 
            pos_dropout=config.decoder_position_dropout, 
            pos_max_len=config.decoder_position_max_len, 
            activation=config.decoder_activation)

        logger.info('Setting encoder/decoder done.')

        # set model & optimizer
        encoder_out_embed_size = config.repr_embed_size
        if config.span_aggregation_choice == 'cat':
            encoder_out_embed_size = config.repr_embed_size*2

        model = Seq2seqModel(
            word_embed_size=config.word_embed_size,
            vocab_size=len(tokenizer),
            encoder=encoder,
            encoder_out_embed_size=encoder_out_embed_size,
            window_size=config.window_size,
            slide_step=config.slide_step,
            decoder=decoder,
            decoder_in_embed_size=config.repr_embed_size,
            device=device,
            span_aggregation_choice=config.span_aggregation_choice,
            masking_ratio=config.masking_ratio_levels[config.cur_masking_ratio_level],
            masking_weight=config.masking_weight_levels[config.cur_masking_weight_level],
            logger=logger,
            ).to(device)

        # load checkpoint
        if f'./sum_dist/checkpoint/{args.exp_name}/checkpoint_{checkpoint_idx}.pt' is not None and os.path.exists(f'./sum_dist/checkpoint/{args.exp_name}/checkpoint_{checkpoint_idx}.pt'):
            checkpoint = torch.load(f'./sum_dist/checkpoint/{args.exp_name}/checkpoint_{checkpoint_idx}.pt')

            cur_step = checkpoint['step']
            model.load_state_dict(checkpoint['model'])

            logger.info(f'Load model from: ./sum_dist/checkpoint/{args.exp_name}/checkpoint_{checkpoint_idx}.pt')

        logger.info('Loading model done.')

        # set data loader
        data_loader_test = DataLoader(
            dataset=dataset, 
            batch_size=args.batch_size, 
            collate_fn=collate_fn)

        logger.info('Setting data loader done.')

        # make output dir
        if not os.path.exists(f'./sum_dist/output/inference/{args.exp_name}/checkpoint_{checkpoint_idx}'):
            os.makedirs(f'./sum_dist/output/inference/{args.exp_name}/checkpoint_{checkpoint_idx}')

        # write truncated source
        if args.write_source:
            split = 'test'
            output_truncated_source_filename = f'./sum_dist/output/inference/{args.exp_name}/checkpoint_{checkpoint_idx}/source_truncated-{split}.txt'

            with open(output_truncated_source_filename, 'w') as source_f:
                write_newline = False
                for batch in data_loader_test:
                    for instance_ids in batch['input_ids']:
                        article = tokenizer.decode(instance_ids, skip_special_tokens=True)
                        if write_newline:
                            source_f.write('\n')
                        source_f.write(article)
                        write_newline = True
            del article

        # set trainer
        trainer = Trainer(
            config=config,
            model=model, 
            tokenizer=tokenizer,
            rouge_calculator=rouge_calculator,
            log_dir=f'./sum_dist/logs/inference/{args.exp_name}',
            logger=logger,
            device=device)

        logger.info('Setting trainer done.')
        logger.info('Start inference...')

        # inference
        logger.info('Run test...')

        # calculate decoding result & attn
        test_results, test_attn_results, test_encoder_spans_for_attn = trainer.inference(
            data_loader=data_loader_test, 
            batch_size=args.batch_size,
            max_len=args.decoding_max_len,
            final_decode_len=args.decoding_target_seq_len,
            decode_times=config.decoding_times,
            return_attn=args.run_attn,
            use_high_rouge=args.use_high_rouge,
            use_high_freq=args.use_high_freq,
            similarity_threshold=args.similarity_threshold,
            spacy_model_ver=config.preprocess_spacy_model,
            filter_len=args.filter_len)

        # save decoding results
        output_prediction_filename = f'./sum_dist/output/inference/{args.exp_name}/checkpoint_{checkpoint_idx}/prediction_all-decode{args.decoding_times}-test.txt'
        cleaned_test_results = []
        with open(output_prediction_filename, 'w') as f:
            for pred_ind, pred_article in enumerate(tqdm(test_results)):
                processed_article = pred_article.replace('[CLS]', '')
                processed_article = pred_article.replace('[PAD]', '')
                processed_article = re.sub(' +', ' ', pred_article)
                f.write(processed_article)
                cleaned_test_results.append(processed_article)
                if pred_ind < len(test_results) - 1:
                    f.write('\n')

        if args.run_attn:
            # save attn weight pkl
            pickle_file_path = f'./sum_dist/output/inference/{args.exp_name}/checkpoint_{checkpoint_idx}/attn_weight-decode{args.decoding_times}-test.pkl'
            with open(pickle_file_path, 'wb') as handle:
                pickle.dump({
                    'weights': test_attn_results,
                    'tokens': test_encoder_spans_for_attn,
                }, handle)
            logger.info(f'Saving attention weight value in: {pickle_file_path}')

    return

if __name__ == '__main__':
    main()
