"""
Read target_txt and prediction_txt, 2 text files.
Both containing 1 article per line.
Output ROUGE score to score.txt.
"""

import spacy

import argparse
import logging
import re
import os
import sys
import unicodedata

from sum_dist.utils.evaluate import RougeCalculator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-target_file', default='./sum_dist/output/inference/transformer22/lg/window5_s1-mask-my_loss_masking_pos-stage4-dec2-len100-attn_d_100/target_all-test.txt')
    parser.add_argument('-temp_target_dir', default='./temp/test/gold', help='writing temp files for ROUGE input')
    parser.add_argument('-prediction_file', default='./sum_dist/output/inference/transformer22/lg/window5_s1-mask-my_loss_masking_pos-stage4-dec2-len100-attn_d_100/checkpoint_31398/prediction_all-decode2-test.txt')
    parser.add_argument('-temp_prediction_dir', default='./temp/test/pred', help='writing temp files for ROUGE input')
    parser.add_argument('-output_dir', default='./sum_dist/output/inference/transformer22/lg/window5_s1-mask-my_loss_masking_pos-stage4-dec2-len100-attn_d_100/checkpoint_31398')
    parser.add_argument('-output_filename_prefix', default='score-test-dec2')
    parser.add_argument('-lang', default='en', choices=['en', 'ru'])
    parser.add_argument('-truncate_len', type=int, default=100)
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel("INFO")

    if args.lang == 'ru':
        spacy_model = spacy.load("ru_core_news_sm")

    rouge_calculator = RougeCalculator(
        prediction_dir=args.temp_prediction_dir, 
        gold_dir=args.temp_target_dir, 
        prediction_prefix='prediction', 
        gold_prefix='gold',
        logger=logger)

    ignore_idx = []
    target = []
    prediction = []
    ind = 0
    with open(args.target_file, 'r') as target_f, open(args.prediction_file, 'r') as pred_f:
        while True:
            target_line = target_f.readline()
            prediction_line = pred_f.readline()

            if len(target_line) > 10:
                # process prediction
                prediction_line = prediction_line.strip('\n')
                prediction_line = prediction_line.replace('[CLS]', '')
                prediction_line = prediction_line.replace('[PAD]', '')
                prediction_line = prediction_line.replace('[', ' ') 
                prediction_line = prediction_line.replace(']', ' ')
                prediction_line = prediction_line.replace('<', ' ')
                prediction_line = prediction_line.replace('>', ' ')
                prediction_line = prediction_line.replace('«', ' ')
                prediction_line = prediction_line.replace('»', ' ')
                prediction_line = re.sub(' +', ' ', prediction_line)
                prediction_line = prediction_line.rstrip(' ').lstrip(' ')
                words = prediction_line.split(' ')
                if len(words) > args.truncate_len:
                    prediction_line = ' '.join(words[:args.truncate_len])

                if len(prediction_line) > 10:
                    prediction_line = unicodedata.normalize("NFKD", prediction_line)
                    target_line = unicodedata.normalize("NFKD", target_line)
                    target_line = target_line.replace('<', ' ')
                    target_line = target_line.replace('>', ' ')
                    target_line = target_line.replace('«', ' ')
                    target_line = target_line.replace('»', ' ')
                    target_line = re.sub(' +', ' ', target_line)
                    target_line = target_line.strip('\n').split(' [NEWLINE] ').copy()

                    if args.lang == 'ru':
                        spacy_pred_result = spacy_model(prediction_line)
                        pred_tokens = [token.text for token in spacy_pred_result]

                        spacy_targ_results = [spacy_model(line).copy() for line in target_line]
                        targ_tokens = [[token.text for token in spacy_result] for spacy_result in spacy_targ_results]

                        all_tokens = [token for t in targ_tokens for token in t] + pred_tokens
                        token_set = list(set(all_tokens))
                        token2ids = {token: str(ids) for ids, token in enumerate(token_set)}

                        prediction_line = ' '.join([token2ids[token] for token in pred_tokens])
                        target_line = [' '.join([token2ids[token] for token in t]) for t in targ_tokens]

                    prediction.append(prediction_line)
                    target.append(target_line)

                else:
                    ignore_idx.append(ind)
            else:
                ignore_idx.append(ind)

            ind += 1
            if not target_line or not prediction_line:
                break

    assert len(prediction) == len(target)

    rouge_calculator.convert_article_to_rouge_file(prediction, is_prediction=True)
    rouge_calculator.convert_article_to_rouge_file(target, is_prediction=False, do_sent_tokenize=False)

    score = rouge_calculator.get_score()
    print('ignore indexes:')
    print(ignore_idx)
    print(len(target))
    print(len(prediction))
    print(score)

    if not os.path.exists(f'{args.output_dir}'):
        os.makedirs(f'{args.output_dir}')

    with open(f'{args.output_dir}/{args.output_filename_prefix}-len{args.truncate_len}.txt', 'w') as f:
        f.write(str(score))
        f.write('\n')

    return


if __name__ == '__main__':
    main()
