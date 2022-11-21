import transformers
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk import sent_tokenize


class Tokenizer(object):
    def __init__(self):
        # config
        self.lang = 'english'
        self.bert_version = 'bert-base-uncased'

        # tokenizer
        self.bert_tokenizer = transformers.BertTokenizer.from_pretrained(self.bert_version)
        self.nltk_tokenizer = word_tokenize
        self.nltk_sent_tokenizer = sent_tokenize

        # nltk - stemming
        self.stemmer = SnowballStemmer(self.lang)

        # nltk - detokenizer
        self.nltk_detokenizer = TreebankWordDetokenizer()

        return

    def tokenize(self, article, tokenize_method, do_stemming=True):
        if tokenize_method not in ['space', 'bert', 'nltk']:
            raise ValueError(f'Unexpected value {tokenize_method} for `tokenize_method`, should be [`space`, `bert`, `nltk`].')

        if tokenize_method == 'space':
            tokens = article.split(' ')
        elif tokenize_method == 'bert':
            tokens = self.bert_tokenizer.tokenize(article)
        elif tokenize_method == 'nltk':
            tokens = self.nltk_tokenizer(article)

        if do_stemming:
            tokens = [self.stemmer.stem(token.lower()) for token in tokens]
        else:
            tokens = [token.lower() for token in tokens if token.lower()]

        return tokens

    def tokens_to_string(self, tokens, tokenize_method):
        if tokenize_method == 'bert':
            result = self.bert_tokenizer.convert_tokens_to_string(tokens)
        elif tokenize_method == 'space':
            result = ' '.join(tokens)
        elif tokenize_method == 'nltk':
            result = self.nltk_detokenizer.detokenize(tokens)

        return result

    def sent_tokenize(self, article):
        return self.nltk_sent_tokenizer(article)
