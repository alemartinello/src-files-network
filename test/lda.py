# -*- coding: utf-8 -*-

import gensim
import h5py
import numpy as np
import os
import random
import warnings

from src.fui.ldatools import preprocess
from src.fui.utils import params, read_h5py
from functools import partial
from multiprocessing import Pool

class LDA:
    def __init__(self, lemmatizer, test_share=0.05, test=False):
        self.dictionary = None
        self.articles = []
        self.article_id = []
        self.SerializedCorpus = None
        self.test = test        
        self.lemmatizer = lemmatizer
        self.test_share = test_share
                
        #if params().options['lda']['log']:
        import logging
        try:
            os.remove(params().paths['lda']+'lda_log.txt')
        except (FileNotFoundError, PermissionError):
            pass
        logging.basicConfig(filename=params().paths['lda']+'lda_log.txt',
                            format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
        logger = logging.getLogger(__name__)
        warnings.filterwarnings('ignore', category=DeprecationWarning)

    def __iter__(self):
        for line in self.articles:
            yield self.bigram_phraser[line.split()]

    def load_and_clean_body_text(self):
        print("No existing pre-processed data found. Loading h5-file for preprocessing")

        df = read_h5py(os.path.join(params().paths['parsed_news'],
                                    params().filenames['parsed_news']))

        try:
            self.articles.extend(list(df['body'].values))
            self.article_id.extend(list(df['article_id'].values))
        except KeyError:
            print("File doesn't contain any body-text")

        # Perform LDA on smaller sample, just for efficiency in case of testing...
        if self.test is True:
            random.seed(1)
            test_idx = random.sample(range(0, len(self.articles)), params().options['lda']['test_size'])
            self.articles = [self.articles[i] for i in test_idx]
            self.article_id = [self.article_id[i] for i in test_idx]

        # Pre-process LDA-docs
        if len(self.articles):
            print("\tProcessing {} documents for LDA".format(len(self.articles)))
            with Pool(params().options['threads']) as pool:
                self.articles = pool.map(partial(preprocess, 
                                                 lemmatizer=self.lemmatizer),
                                                 self.articles)

            print("\tSaving cleaned documents")
            folder_path = params().paths['lda']
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            file_path = os.path.join(folder_path, params().filenames['lda_cleaned_text'])
            with h5py.File(file_path, 'w') as hf:
                data = np.array(list(zip(self.article_id, self.articles)), dtype=object)
                string_dt = h5py.string_dtype(encoding='utf-8')
                hf.create_dataset('parsed_strings', data=data, dtype=string_dt)
        # Train bigram model
        self.load_bigrams()

    def load_processed_text(self):
        try:
            with h5py.File(os.path.join(params().paths['lda'], params().filenames['lda_cleaned_text']), 'r') as hf:
                print("Loading processed data from HDF-file")
                hf = hf['parsed_strings'][:]
                self.article_id = list(zip(*hf))[0]
                self.articles = list(zip(*hf))[1]
                print("\t{} documents loaded".format(len(self.articles)))
            return 1
        except OSError:
            return 0
    
    def load_bigrams(self):
        if os.path.isfile(os.path.join(params().paths['lda'],'phrases.pkl')):
            phrases = gensim.utils.SaveLoad.load(os.path.join(params().paths['lda'],'phrases.pkl'))
            self.bigram_phraser = gensim.models.phrases.Phraser(phrases)
            print("Bigram phraser loaded")
        else:
            print("Bigram phraser not found, training")
            with h5py.File(os.path.join(params().paths['lda'], params().filenames['lda_cleaned_text']), 'r') as hf:
                hf = hf['parsed_strings'][:]
                articles_to_phrasing  = [a[1].split() for a in hf]
            phrases = gensim.models.phrases.Phrases(articles_to_phrasing, params().options['lda']['no_below'], threshold=100)
            phrases.save(os.path.join(params().paths['lda'],'phrases.pkl'), separately=None, sep_limit=10485760, ignore=frozenset([]), pickle_protocol=2)
            self.bigram_phraser = gensim.models.phrases.Phraser(phrases)
            print("Bigram phraser loaded")
        
    def get_topics(self, lda_model, dictionary, text):
        text = preprocess(text, self.lemmatizer)
        bow = dictionary.doc2bow(self.bigram_phraser[text.split()])
        return lda_model.get_document_topics(bow, minimum_probability=0.0)

