"""
Load models of the given exp(dir) & checkpoint number. 
Print out the number of parameter of the loaded model.
"""

from transformers import BertTokenizer
import torch

import argparse
import os

from sum_dist.configs import MConfigs
from sum_dist.models.encoder import TransformerEncoder
from sum_dist.models.decoder import TransformerDecoder
from sum_dist.models.seq2seq import Seq2seqModel
import sum_dist.utils.logging as logging


logger = logging.get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()

    # run time settings
    parser.add_argument('-exp_name', type=str, nargs='?', default='transformer22/lg/window5_s1-mask-my_loss_masking_pos')
    parser.add_argument('-checkpoint_idx', type=str, nargs='?', default='17941')
    parser.add_argument('-decoding_max_len', type=int, nargs='?', default=500)
    parser.add_argument('-decoding_target_seq_len', type=int, nargs='?', default=100)
    parser.add_argument('-device', type=str, nargs='?', default='cuda:0')

    # training paths
    parser.add_argument('-load_config_dir', type=str, nargs='?', default=None)

    args = parser.parse_args()
    logger.info(args)

    device = torch.device(args.device)
    logger.info(f'Using device: {device}')

    config = MConfigs()
    tokenizer = None
    encoder = None
    decoder = None

    # load config
    if f'./sum_dist/checkpoint/{args.exp_name}/checkpoint_{args.checkpoint_idx}.pt' is not None and os.path.exists(f'./sum_dist/checkpoint/{args.exp_name}/checkpoint_{args.checkpoint_idx}.pt'):
        checkpoint = torch.load(f'./sum_dist/checkpoint/{args.exp_name}/checkpoint_{args.checkpoint_idx}.pt')
        config = checkpoint['config']
        logger.info(f'Load checkpoint config from: ./sum_dist/checkpoint/{args.exp_name}/checkpoint_{args.checkpoint_idx}.pt')
    
    if args.load_config_dir is not None and os.path.exists(args.load_config_dir):
        config = config.load_json(load_dir=args.load_config_dir)
        logger.info(f'Load json config from: {args.load_config_dir}')

    logger.info(config.__dict__)

    # set encoder
    tokenizer = BertTokenizer.from_pretrained(config.bert_version)
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

    # load checkpoint
    if f'./sum_dist/checkpoint/{args.exp_name}/checkpoint_{args.checkpoint_idx}.pt' is not None and os.path.exists(f'./sum_dist/checkpoint/{args.exp_name}/checkpoint_{args.checkpoint_idx}.pt'):
        checkpoint = torch.load(f'./sum_dist/checkpoint/{args.exp_name}/checkpoint_{args.checkpoint_idx}.pt')
        model.load_state_dict(checkpoint['model'])

        logger.info(f'Load model from: ./sum_dist/checkpoint/{args.exp_name}/checkpoint_{args.checkpoint_idx}.pt')

    logger.info('Loading model done.')

    # count model param
    print(f'model param: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
        
    return

if __name__ == '__main__':
    main()
