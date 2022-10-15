import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BertTokenizer

import argparse
import os
import random

from sum_dist.configs import MConfigs
from sum_dist.models.encoder import TransformerEncoder
from sum_dist.models.decoder import TransformerDecoder
from sum_dist.models.seq2seq import Seq2seqModel
from sum_dist.models.loss import ReconstructionLoss
from sum_dist.trainer import Trainer
from sum_dist.utils.data.collate_fn import InitCollate
from sum_dist.utils.data.cnndm import DatasetCNNDM
from sum_dist.utils.data.xsum import DatasetXSUM
from sum_dist.utils.data.mlsum import DatasetMLSUMde, DatasetMLSUMes, DatasetMLSUMru
from sum_dist.utils.data.arxiv import DatasetArxiv
from sum_dist.utils.data.wiki import DatasetWiki
from sum_dist.utils.evaluate import RougeCalculator
import sum_dist.utils.logging as logging
from sum_dist.utils.parse import str2bool


logger = logging.get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()

    # training setting
    parser.add_argument('-mode', type=str, nargs='?', default='normal', choices=['normal', 'order'])
    parser.add_argument('-dataset', type=str, nargs='?', default='cnndm', choices=['cnndm', 'xsum', 'mlsum_de', 'mlsum_es', 'mlsum_ru', 'wiki_en', 'arxiv'])
    parser.add_argument('-start_batch_idx', type=int, nargs='?', default=0)
    parser.add_argument('-batch_size_train', type=int, nargs='?', default=4)
    parser.add_argument('-accumulation_step', type=int, nargs='?', default=64)
    parser.add_argument('-save_checkpoint_step', type=int, nargs='?', default=200) # save every `save_checkpoint_step*accumulation_step`
    parser.add_argument('-device', type=str, nargs='?', default='cuda:0')

    # training path
    parser.add_argument('-log_dir', type=str, nargs='?', default='./sum_dist/logs/train/transformer22/lg/window5_s1-mask-my_loss_masking_pos-xsum')
    parser.add_argument('-load_config_dir', type=str, nargs='?', default='./sum_dist/exp_configs/0030-1.json')
    parser.add_argument('-load_checkpoint_dir', type=str, nargs='?', default=None)
    parser.add_argument('-save_checkpoint_dir', type=str, nargs='?', default='./sum_dist/checkpoint/transformer22/lg/window5_s1-mask-my_loss_masking_pos-xsum')

    # rouge path
    parser.add_argument('-prediction_dest', type=str, nargs='?', default='./sum_dist/output/transformer22-prediction/lg/window5_s1-mask-my_loss_masking_pos-xsum')
    parser.add_argument('-target_dest', type=str, nargs='?', default='./sum_dist/output/transformer22-gold/lg/window5_s1-mask-my_loss_masking_pos-xsum')
    parser.add_argument('-prediction_file_prefix', type=str, nargs='?', default='prediction')
    parser.add_argument('-target_file_prefix', type=str, nargs='?', default='gold')

    # ann dataset path
    parser.add_argument('-cnn_ann_pkl_dir', type=str, nargs='?', default='./sum_dist/data/preprocess/cnndm-bert-ann.pkl')
    parser.add_argument('-xsum_ann_pkl_dir', type=str, nargs='?', default='./sum_dist/data/preprocess/xsum-bert-ann.pkl')
    parser.add_argument('-mlsum_de_ann_pkl_dir', type=str, nargs='?', default='./sum_dist/data/preprocess/mlsum_de-bert-ann.pkl')
    parser.add_argument('-mlsum_es_ann_pkl_dir', type=str, nargs='?', default='./sum_dist/data/preprocess/mlsum_es-bert-ann.pkl')
    parser.add_argument('-mlsum_ru_ann_pkl_dir', type=str, nargs='?', default='./sum_dist/data/preprocess/mlsum_ru-bert-ann.pkl')
    parser.add_argument('-arxiv_dataset_dir', type=str, nargs='?', default='./sum_dist/data/arxiv_data/arxiv-dataset/arxiv-dataset')
    parser.add_argument('-arxiv_ann_pkl_dir', type=str, nargs='?', default='./sum_dist/data/preprocess/arxiv-bert-ann.pkl')
    parser.add_argument('-wiki_en_ann_pkl_dir', type=str, nargs='?', default='./sum_dist/data/preprocess/wiki_en-bert-ann.pkl')

    # other
    parser.add_argument('-print_detail', type=str2bool, nargs='?', const=True, default=False)

    args = parser.parse_args()
    logger.info(args)

    device = torch.device(args.device)
    config = MConfigs()
    tokenizer = None
    collate_fn_train = None
    encoder = None
    decoder = None

    # load config
    if args.load_checkpoint_dir is not None and os.path.exists(args.load_checkpoint_dir):
        checkpoint = torch.load(args.load_checkpoint_dir)
        config = checkpoint['config']
        logger.info(f'Load checkpoint config from: {args.load_checkpoint_dir}')
    
    if args.load_config_dir is not None and os.path.exists(args.load_config_dir):
        config = config.load_json(load_dir=args.load_config_dir)
        logger.info(f'Load json config from: {args.load_config_dir}')
    
    logger.info(config.__dict__)

    # set seed
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # set tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.bert_version)

    # set collate_fn
    clean_brackets = False
    if args.dataset == 'xsum':
        clean_brackets = True

    collate_fn_train = InitCollate(
        tokenizer=tokenizer, 
        encoder_max_seq_len=config.seq_len,
        decoder_max_seq_len=config.decoder_position_max_len,
        encoder_sampling_len_levels=config.sampling_length_levels,
        cur_encoder_sampling_len_level=config.cur_sampling_length_level,
        clean_brackets=clean_brackets)

    # set encoder
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
        activation=config.decoder_activation
    )

    logger.info('Setting encoder/decoder done.')

    # set dataset
    dataset = {}
    if args.dataset == 'cnndm':
        for split in ['train', 'validation', 'test']:
            num_data = config.num_data[split]
            dataset[split] = DatasetCNNDM(
                dataset_pkl_path=None,
                ann_pkl_path=args.cnn_ann_pkl_dir, 
                num_data=num_data,
                split=split)
    elif args.dataset == 'xsum':
        for split in ['train', 'validation', 'test']:
            num_data = config.num_data[split]
            dataset[split] = DatasetXSUM(
                dataset_pkl_path=None,
                ann_pkl_path=args.xsum_ann_pkl_dir,
                num_data=num_data,
                split=split)
    elif args.dataset == 'mlsum_de':
        for split in ['train', 'validation', 'test']:
            num_data = config.num_data[split]
            dataset[split] = DatasetMLSUMde(
                dataset_pkl_path=None,
                ann_pkl_path=args.mlsum_de_ann_pkl_dir,
                num_data=num_data,
                split=split)
    elif args.dataset == 'mlsum_es':
        for split in ['train', 'validation', 'test']:
            num_data = config.num_data[split]
            dataset[split] = DatasetMLSUMes(
                dataset_pkl_path=None,
                ann_pkl_path=args.mlsum_es_ann_pkl_dir,
                num_data=num_data,
                split=split)
    elif args.dataset == 'mlsum_ru':
        for split in ['train', 'validation', 'test']:
            num_data = config.num_data[split]
            dataset[split] = DatasetMLSUMru(
                dataset_pkl_path=None,
                ann_pkl_path=args.mlsum_ru_ann_pkl_dir,
                num_data=num_data,
                split=split)
    elif args.dataset == 'arxiv':
        for split in ['train', 'validation', 'test']:
            num_data = config.num_data[split]
            dataset[split] = DatasetArxiv(
                dataset_dir=args.arxiv_dataset_dir,
                ann_pkl_path=args.arxiv_ann_pkl_dir,
                num_data=num_data,
                split=split)
    elif args.dataset == 'wiki_en':
        for split in ['train']:
            num_data = config.num_data[split]
            dataset[split] = DatasetWiki(
                dataset_pkl_path=None,
                ann_pkl_path=args.wiki_en_ann_pkl_dir,
                num_data=num_data,
                split=split)

    logger.info(f'Loading dataset done:')
    for split in ['train', 'validation', 'test']:
        if split in dataset.keys():
            logger.info(f'{split}: {len(dataset[split])} data')

    # set model & optimizer
    encoder_out_embed_size = config.repr_embed_size
    if config.span_aggregation_choice == 'cat':
        encoder_out_embed_size = config.repr_embed_size*2

    model = Seq2seqModel(
        word_embed_size=config.word_embed_size,
        vocab_size=len(tokenizer),
        encoder = encoder,
        encoder_out_embed_size=encoder_out_embed_size,
        window_size=config.window_size,
        slide_step=config.slide_step,
        decoder = decoder,
        decoder_in_embed_size=config.repr_embed_size,
        device=device,
        span_aggregation_choice=config.span_aggregation_choice,
        masking_ratio=config.masking_ratio_levels[config.cur_masking_ratio_level],
        masking_weight=config.masking_weight_levels[config.cur_masking_weight_level],
        logger=logger,
        ).to(device)

    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # load checkpoint
    cur_step = 0
    if args.load_checkpoint_dir is not None and os.path.exists(args.load_checkpoint_dir):
        checkpoint = torch.load(args.load_checkpoint_dir)

        cur_step = checkpoint['step']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        logger.info(f'Load model/optimizer config from: {args.load_checkpoint_dir}')

    logger.info('Loading model/optimizer done.')

    # set rouge calculator
    rouge_calculator = RougeCalculator(
        prediction_dir=args.prediction_dest, 
        gold_dir=args.target_dest, 
        prediction_prefix=args.prediction_file_prefix, 
        gold_prefix=args.target_file_prefix,
        logger=logger)

    logger.info('Setting ROUGE calculator done.')

    # set loss
    loss_fn = ReconstructionLoss(
        pad_idx=tokenizer.convert_tokens_to_ids([config.pad_token])[0], 
        ).to(device)

    logger.info('Setting loss fn done.')

    # set trainer
    trainer = Trainer(
        config=config,
        model=model, 
        tokenizer=tokenizer,
        rouge_calculator=rouge_calculator,
        optimizer=optimizer,
        log_dir=args.log_dir,
        device=device,
        logger=logger)

    logger.info('Setting trainer done.')
    logger.info('Training...')

    # set data loader
    data_loader_train = DataLoader(
        dataset=dataset['train'], 
        batch_size=args.batch_size_train, 
        collate_fn=collate_fn_train, 
        drop_last=True,
        shuffle=True)

    # train
    trainer.train(
        mode=args.mode,
        epoch_num=config.epoch_num, 
        start_batch_idx=args.start_batch_idx,
        loss_fn=loss_fn, 
        data_loader_train=data_loader_train,
        collate_fn=collate_fn_train,
        accumulation_step=args.accumulation_step, 
        save_checkpoint_step=args.save_checkpoint_step, 
        save_checkpoint_dir=args.save_checkpoint_dir,
        prediction_dest=args.prediction_dest,
        gold_dest=args.target_dest,
        cur_step=cur_step, 
        print_detail=args.print_detail)

    return


if __name__ == '__main__':
    main()
