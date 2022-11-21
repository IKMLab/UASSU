import argparse
import json
import os
import pickle

import sum_dist.utils.logging as logging

logger = logging.get_logger(__name__)

class MConfigs(object):
    def __init__(self, **kwargs):

        # preprocess args
        self.preprocess_spacy_model = kwargs.pop("preprocess_spacy_model", "en_core_web_sm")

        # encoder args
        self.bert_version = kwargs.pop("bert_version", "bert-base-uncased")
        self.seq_len = kwargs.pop("seq_len", 512)
        self.word_embed_size = kwargs.pop("word_embed_size", 512)
        self.repr_embed_size = kwargs.pop("repr_embed_size", 512)
        self.encoder_num_head = kwargs.pop("encoder_num_head", 8)
        self.encoder_num_layer = kwargs.pop("encoder_num_layer", 2)
        self.encoder_dropout = kwargs.pop("encoder_dropout", 0.1)
        self.encoder_ff_embed_size = kwargs.pop("encoder_ff_embed_size", 1024)
        self.encoder_activation = kwargs.pop("encoder_activation", "relu")

        # decoder args
        self.decoder_num_head = kwargs.pop("decoder_num_head", 8)
        self.decoder_num_layer = kwargs.pop("decoder_num_layer", 2)
        self.decoder_dropout = kwargs.pop("decoder_dropout", 0.1)
        self.decoder_ff_embed_size = kwargs.pop("decoder_ff_embed_size", 1024)
        self.decoder_activation = kwargs.pop("decoder_activation", "relu")
        self.decoder_position_dropout = kwargs.pop("decoder_position_dropout", 0.1)
        self.decoder_position_max_len = kwargs.pop("decoder_position_max_len", 512)

        # masking args
        self.window_size = kwargs.pop("window_size", 5)
        self.slide_step = kwargs.pop("slide_step", 1)
        self.span_aggregation_choice = kwargs.pop("span_aggregation_choice", "avg")

        # training data args
        self.sampling_length_level_threshold = kwargs.pop("sampling_length_level_threshold", 100)
        self.sampling_length_levels = kwargs.pop("sampling_length_levels", [500, 400, 300, 200, 150, 100, 75, 50, 30, 15])
        self.cur_sampling_length_level = kwargs.pop("cur_sampling_length_level", 0)
        self.masking_ratio_level_threshold = kwargs.pop("masking_ratio_level_threshold", 100)
        self.masking_ratio_levels = kwargs.pop("masking_ratio_levels", [0.15, 0.30, 0.45, 0.50, 0.65, 0.80, 0.90])
        self.cur_masking_ratio_level = kwargs.pop("cur_masking_ratio_level", 0)
        self.masking_weight_level_threshold = kwargs.pop("masking_weight_level_threshold", 100)
        self.masking_weight_levels = kwargs.pop("masking_weight_levels", [1.0, 0.5, 1e-1, 5e-2, 1e-2, 1e-3, 1e-4])
        self.cur_masking_weight_level = kwargs.pop("cur_masking_weight_level", 0)
        self.loss_weight_levels_threshold = kwargs.pop("loss_weight_levels_threshold", 100)
        self.loss_weight_levels = kwargs.pop("loss_weight_levels", [1e+2, 1e+1, 1.0, 1e-2, 1e-3, 1e-4, 1e-5])
        self.cur_loss_weight_level = kwargs.pop("cur_loss_weight_level", 0)
        self.num_data = kwargs.pop("num_data", {
            "train": 287113,
            "validation": 13368,
            "test": 11490,
        })

        # training args
        self.epoch_num = kwargs.pop("epoch_num", 5)
        self.lr = kwargs.pop("lr", 2e-5)
        self.grad_clip = kwargs.pop("grad_clip", 5)
        self.weight_decay = kwargs.pop("weight_decay", 0)

        # decoding args
        self.decoding_methods = kwargs.pop("decoding_methods", ['top-k'])
        self.decoding_target_seq_len = kwargs.pop("decoding_target_seq_len", 56)
        self.decoding_times = kwargs.pop("decoding_times", 1)
        self.k = kwargs.pop("k", 30)

        # other args
        self.bos_token = kwargs.pop("bos_token", "[CLS]")
        self.eos_token = kwargs.pop("eos_token", "[SEP]")
        self.pad_token = kwargs.pop("pad_token", "[PAD]")
        self.seed = kwargs.pop("seed", 37)

        # additional args
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f'Cannot set `{key}` with `{value}` for `{self}`.')
                raise err

        return
    
    @classmethod
    def load_json(cls, load_dir):
        if not os.path.exists(load_dir):
            raise FileNotFoundError(f'Path `load_dir` not exists: `{load_dir}`.')

        with open(load_dir, 'r') as f:
            config = json.load(f)
        
        return cls(**config)

    @classmethod
    def load_pkl(cls, load_dir):
        if not os.path.exists(load_dir):
            raise FileNotFoundError(f'Path `load_dir` not exists: `{load_dir}`.')

        with open(load_dir, 'rb') as f:
            config = pickle.load(f)

        return cls(**config)
    
    def save_json(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(f'{save_dir}/config.json', 'w') as f:
            json.dump(self.__dict__, f)

    def save_pkl(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(f'{save_dir}/config.pkl', 'wb') as f:
            pickle.dump(self.__dict__, f)

    def update(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)
