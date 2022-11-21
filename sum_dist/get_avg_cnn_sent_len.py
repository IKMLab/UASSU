"""
Load CNNDM dataset & count the number of tokens in each clause.
Print out the average number of tokens in a clause.
"""

import transformers
from tqdm import tqdm
from sum_dist.utils.data.cnndm import DatasetCNNDM
import re

def main():
    dataset = DatasetCNNDM(
        dataset_pkl_path=None,
        ann_pkl_path='./sum_dist/data/preprocess/cnndm-bert-ann.pkl', 
        split='train')
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    sent_lens = []
    for instance in tqdm(dataset):
        cur_line = instance['article'].strip('\n')

        cur_sentences = re.split('[!?.,]', cur_line)

        for sent in cur_sentences:
            tokens = tokenizer.tokenize(sent)
            sent_lens.append(len(tokens))

    print(sum(sent_lens)/len(sent_lens))

    return

if __name__ == '__main__':
    main()
