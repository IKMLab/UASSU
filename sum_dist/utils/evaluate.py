import nltk
from pyrouge import Rouge155

import os

class RougeCalculator(object):

    def __init__(self, prediction_dir, gold_dir, prediction_prefix, gold_prefix, logger=None):
        self.logger = logger
        self.rouge = Rouge155()
        self.set_prediction_dir(prediction_dir)
        self.set_gold_dir(gold_dir)
        self.set_prediction_prefix(prediction_prefix)
        self.set_gold_prefix(gold_prefix)
        return


    def convert_article_to_rouge_file(self, articles, is_prediction=True, prediction_dir=None, gold_dir=None, do_sent_tokenize=True):
        """
        Convert a list of articles to files which ROUGE understands.
        Args:
            - `articles` (List[str])
        """
        if not isinstance(articles, list):
            raise ValueError(f'Expect arg `articles`: `{articles}` to be type `list`.')

        if not isinstance(is_prediction, bool):
            raise ValueError(f'Expect arg `is_prediction`: `{is_prediction}` to be type `list`.')

        if is_prediction:
            file_prefix = self.system_prefix
            if prediction_dir is not None:
                self.set_prediction_dir(prediction_dir)
                dest_dir = prediction_dir
            else:
                dest_dir = self.rouge.system_dir
        else:
            file_prefix = self.model_prefix
            if gold_dir is not None:
                self.set_gold_dir(gold_dir)
                dest_dir = gold_dir
            else:
                dest_dir = self.rouge.model_dir

        width = len(str(len(articles)))

        for index, article in enumerate(articles):
            if isinstance(article, str):
                cur_article = article.lower()
            elif isinstance(article, list):
                cur_article = [sent.lower() for sent in article]

            if do_sent_tokenize:
                sentences = nltk.sent_tokenize(cur_article)
            else:
                sentences = cur_article

            with open(f'{dest_dir}/{file_prefix}_{index:0{width}}.txt', 'w') as f:
                for sent_index, sentence in enumerate(sentences):
                    f.write(sentence)
                    if sent_index < len(sentences):
                        f.write('\n')

        return


    def get_score(self, prediction_dir=None, gold_dir=None, print_detail=False):
        if prediction_dir is not None:
            self.set_prediction_dir(prediction_dir)
        if gold_dir is not None:
            self.set_gold_dir(gold_dir)
        if self.logger is not None:
            self.logger.info(f'Calculating rouge score between directories:')
            self.logger.info(f'- prediction_dir: {self.rouge.system_dir}')
            self.logger.info(f'- gold_dir: {self.rouge.model_dir}')
        output = self.rouge.convert_and_evaluate()
        return self.rouge.output_to_dict(output)


    def set_prediction_dir(self, prediction_dir):
        if self.rouge is None:
            raise NameError(f'Variable `self.rouge` has not been initiated.')

        if not isinstance(prediction_dir, str):
            raise ValueError(f'Expect parameter `prediction_dir`: `{prediction_dir}` to be type `str`.')

        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir)

        self.rouge.system_dir = prediction_dir

        if self.logger is not None:
            self.logger.info(f'In evaluate.py - class RougeCalculator: Setting ROUGE `prediction_dir` to `{self.rouge.system_dir}`.')
        return

    
    def set_gold_dir(self, gold_dir):
        if self.rouge is None:
            raise NameError(f'Variable `self.rouge` has not been initiated.')

        if not isinstance(gold_dir, str):
            raise ValueError(f'Expect parameter `gold_dir`: `{gold_dir}` to be type `str`.')
        
        if not os.path.exists(gold_dir):
            os.makedirs(gold_dir)

        self.rouge.model_dir = gold_dir

        if self.logger is not None:
            self.logger.info(f'In evaluate.py - class RougeCalculator: Setting ROUGE `gold_dir` to `{self.rouge.model_dir}`.')
        return


    def set_prediction_prefix(self, prediction_prefix):
        if self.rouge is None:
            raise NameError(f'Variable `self.rouge` has not been initiated.')

        if not isinstance(prediction_prefix, str):
            raise ValueError(f'Expect parameter `prediction_prefix`: `{prediction_prefix}` to be type `str`.')

        self.system_prefix = prediction_prefix
        self.rouge.system_filename_pattern = f'{prediction_prefix}_(\d+).txt'
        return


    def set_gold_prefix(self, gold_prefix):
        if self.rouge is None:
            raise NameError(f'Variable `self.rouge` has not been initiated.')

        if not isinstance(gold_prefix, str):
            raise ValueError(f'Expect parameter `gold_prefix`: `{gold_prefix}` to be type `str`.')
        
        self.model_prefix = gold_prefix
        self.rouge.model_filename_pattern = f'{gold_prefix}_#ID#.txt'
        return


