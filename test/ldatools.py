import gensim
#from gensim.models.callbacks import CoherenceMetric, PerplexityMetric, ConvergenceMetric
import numpy as np
import os
import pandas as pd
import random
import csv
import json
import copy
import codecs

from collections import Counter
from functools import partial
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from multiprocessing import Pool
from wordcloud import WordCloud
from langdetect import detect

from src.fui.utils import timestamp, params, read_h5py


def __remove_stopwords(word_list, stopfile):
    if not os.path.exists(stopfile):
        raise Exception('No stopword file in directory')
    stopwords_file = open(stopfile, "r")
    stopwords = stopwords_file.read().splitlines()
    word_list = [word for word in word_list if word not in stopwords]
    return word_list   

def preprocess(text, lemmatizer, stopfile='../data/stopwords.txt'):
    """
    - simple_preprocess (convert all words to lowercase, remove punctuations, floats, newlines (\n),
    tabs (\t) and split to words)
    - remove all stopwords
    - remove words with a length < threshold
    - lemmatize
    """
    error=['år','bankernes','bankerne', 'dst','priser','bankers']
    drop=['bre','nam','ritzau','st','le','sin','år','stor','me','når','se','dag','en','to','tre','fire','fem','seks','syv','otte','ni','ti']
    if detect(text) == 'en':
        return ''
    
    text = gensim.utils.simple_preprocess(text, deacc=False, max_len=25)
    list_to_stem = __remove_stopwords(text, stopfile)
        
    lemmed_list = [lemmatizer.lemmatize("", word)[0] for word in list_to_stem if not word in error]
    tokens = [word for word in lemmed_list if len(word) >= 2]
    
    tokens = [w for w in tokens if not w in drop]
    tokens = [w.replace("bankernes","bank") for w in tokens ]
    tokens = [w.replace("bankers","bank") for w in tokens ]
    tokens = [w.replace("kris","krise") for w in tokens ]
    tokens = [w.replace("bile","bil") for w in tokens ]
    tokens = [w.replace("bankerne","bank") for w in tokens ]
    tokens = [w.replace("priser","pris") for w in tokens ]
    tokens=[w for w in tokens if not w.isdigit()]
    
    return ' '.join([word for word in tokens])

def print_topics(lda_instance, topn=30, unique_sort=True):
    lda_model = lda_instance.lda_model
    
    csv_path = os.path.join(params().paths['lda'], 
                            'topic_words'+str(lda_model.num_topics)+'.csv') 
    header = ['topic_'+str(x) for x in range(lda_model.num_topics)]
    
    if not unique_sort:
        word_lists = []
        for t in range(lda_model.num_topics):
            word_list = lda_model.show_topic(t,topn)
            if not len(word_lists):
                word_list = [[w[0]] for w in word_list]
                word_lists = word_list
            else:
                word_list = [w[0] for w in word_list]
                for i in range(topn):
                    word_lists[i].append(word_list[i])
        with open(csv_path, mode='w', newline='\n', encoding='utf-8-sig') as csv_out:
            csvwriter = csv.writer(csv_out, delimiter=',')
            csvwriter.writerow(header)
            for i in range(topn):
                csvwriter.writerow(word_lists[i])
        return word_lists
        
    else: 
        df = get_unique_words(lda_instance, topn)

        df = df[['word']]
        df.index = pd.MultiIndex.from_arrays(
            [df.index.get_level_values(1), df.groupby(level=1).cumcount()],
            names=['token', 'topic'])
        df = df.unstack(level=0)
        df.to_csv(csv_path,header=header,encoding='utf-8-sig',index=False)
        return df
        
def optimize_topics(lda_instance, topics_to_optimize, plot=False, plot_title=""):
    coherence_scores = []
    lda_models = []

    print("Finding coherence-scores for the list {}:".format(topics_to_optimize))
    for num_topics in topics_to_optimize:
        print("\t{} topics... {}".format(num_topics, timestamp()))

        lda_model_n = gensim.models.LdaMulticore(corpus=lda_instance.TrainCorpus,
                                                 num_topics=num_topics,
                                                 id2word=lda_instance.dictionary,
                                                 passes=20, per_word_topics=False,
                                                 alpha='asymmetric',
                                                 eval_every=100,
                                                 minimum_probability=0.0,
                                                 chunksize=10000, workers=16)

        coherence_model_n = gensim.models.CoherenceModel(model=lda_model_n,
                                                         texts=(articles for articles in lda_instance),
                                                         dictionary=lda_instance.dictionary,
                                                         coherence='c_v',
                                                         processes=16)
        lda_models.append(lda_model_n)
        coherence_scores.append(coherence_model_n.get_coherence())

        try:
            folder_path = os.path.join(params().paths['lda'], 'lda_model_' + str(num_topics))
            file_path = os.path.join(folder_path, 'trained_lda')
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            lda_model_n.save(file_path)
            print("LDA-model saved ({} topics)".format(num_topics))
        except FileNotFoundError:
            print("Error: LDA-file not found")

        with open('coherence.csv', 'a+', newline='') as csvout:
            wr = csv.writer(csvout, delimiter=',', lineterminator='\n')
            wr.writerow([num_topics, coherence_model_n.get_coherence()])

    for n, cv in zip(topics_to_optimize, coherence_scores):
        print("LDA with {} topics has a coherence-score {}".format(n, round(cv, 2)))
        


    if plot:
        plt.plot(topics_to_optimize, coherence_scores)
        plt.xlabel('Number of topics')
        plt.ylabel('Coherence score')
        #plot_title += str(params().options['lda']['tf-idf'])
        #plt.title(plot_title)
        plt.show()

    return lda_models, coherence_scores

def create_dictionary(lda_instance, load_bigrams=True, unwanted_words=None, keep_words=None):
    
    # Clean and write texts to HDF
    if not lda_instance.load_processed_text():
        lda_instance.load_and_clean_body_text()
        
    # Create dictionary (id2word)
    file_path = os.path.join(params().paths['lda'], params().filenames['lda_dictionary'])
    
    # Load bigram phraser
    if load_bigrams:
        lda_instance.load_bigrams()
            
    try:
        lda_instance.dictionary = gensim.corpora.Dictionary.load(file_path)
        print("Loaded pre-existing dictionary")
    except FileNotFoundError:
        print("Dictionary not found, creating from scratch")

        lda_instance.dictionary = gensim.corpora.Dictionary(articles for articles in lda_instance)

        lda_instance.dictionary.filter_extremes(no_below=params().options['lda']['no_below'],
                                                no_above=params().options['lda']['no_above'],
                                                keep_n=params().options['lda']['keep_n'],
                                                keep_tokens=keep_words)
        if unwanted_words is None:
            unwanted_words = []
        unwanted_ids = [k for k, v in lda_instance.dictionary.items() if v in unwanted_words]
        lda_instance.dictionary.filter_tokens(bad_ids=unwanted_ids)
        lda_instance.dictionary.compactify()
        lda_instance.dictionary.save(file_path)
    print("\t{}".format(lda_instance.dictionary))
       
def create_corpus(lda_instance):
    
    # Helper-class to create BoW-corpus "lazily"
    class CorpusSplitter:
        def __init__(self, test_share):
            self.test_share = test_share
            self.test_corpus = []
        
        def __iter__(self):
            for line in lda_instance.articles:
                if random.random() <= self.test_share:
                    self.test_corpus.append(lda_instance.dictionary.doc2bow(lda_instance.bigram_phraser[line.split()]))
                    continue
                else:
                    yield lda_instance.dictionary.doc2bow(lda_instance.bigram_phraser[line.split()])

    # Serialize corpus using either BoW of tf-idf
    corpus_bow = CorpusSplitter(lda_instance.test_share)
    
    file_path = os.path.join(params().paths['lda'], 'corpus.mm')
    file_path_test = os.path.join(params().paths['lda'], 'corpus_test.mm')
    try:
        lda_instance.TrainCorpus = gensim.corpora.MmCorpus(file_path)
        if lda_instance.test_share > 0.0:
            lda_instance.TestCorpus = gensim.corpora.MmCorpus(file_path_test)
        print("Loaded pre-existing corpus")
    except FileNotFoundError:
        print("Corpus not found, creating from scratch")
        if not hasattr(lda_instance, 'bigram_phraser'):
            lda_instance.load_bigrams()

        # Serialize corpus (either BoW or tf-idf)
        if not params().options['lda']['tf-idf']:
            print("\tSerializing corpus, BoW")
            gensim.corpora.MmCorpus.serialize(file_path, corpus_bow)
            if lda_instance.test_share > 0.0:
                gensim.corpora.MmCorpus.serialize(file_path_test, corpus_bow.test_corpus)
        else:
            print("\tSerializing corpus, tf-idf")
            tfidf = gensim.models.TfidfModel(corpus_bow)
            train_corpus_tfidf = tfidf[corpus_bow]
            gensim.corpora.MmCorpus.serialize(file_path, train_corpus_tfidf)
            if lda_instance.test_share > 0.0:
                tfidf = gensim.models.TfidfModel(corpus_bow.test_corpus)
                test_corpus_tfidf = tfidf[corpus_bow.test_corpus]
                gensim.corpora.MmCorpus.serialize(file_path_test, test_corpus_tfidf)

        lda_instance.TrainCorpus = gensim.corpora.MmCorpus(file_path)
        if lda_instance.test_share > 0.0:
            lda_instance.TestCorpus = gensim.corpora.MmCorpus(file_path_test)

def save_models(lda_instance):

    # Save all models in their respective folder
    for i, lda_model in enumerate(lda_instance.lda_models):
        try:
            folder_path = os.path.join(params().paths['lda'], 'lda_model_' + str(lda_model.num_topics))
            file_path = os.path.join(folder_path, 'trained_lda')
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            lda_model.save(file_path)
            print("LDA-model #{} saved ({} topics)".format(i, lda_model.num_topics))
        except FileNotFoundError:
            print("Error: LDA-file not found")
        except IndexError:
            print("Error: List index out of range")


def load_model(lda_instance, num_topics):
    try:
        folder_path = os.path.join(params().paths['root'],params().paths['lda'], 'lda_model_' + str(num_topics))
        file_path = os.path.join(folder_path, 'trained_lda')
        lda_instance.lda_model = gensim.models.LdaMulticore.load(file_path)
        print("LDA-model with {} topics loaded".format(num_topics))
    except FileNotFoundError:
        print("Error: LDA-model not found")
        lda_instance.lda_model = None
        
def load_models(lda_instance, topics, plot=False):
    lda_models = []
    file_list = []
    for t in topics: 
        print(t)
        file_list.append(os.path.join(params().paths['root'],params().paths['lda'], 'lda_model_'+str(t)+'\\trained_lda'))


    for f in file_list:
        print(f)
        try:
            lda_model_n = gensim.models.LdaMulticore.load(f)
            lda_models.append(lda_model_n)

        except FileNotFoundError:
            print(f"Error: LDA-model at {f} not found")

    lda_instance.lda_models = lda_models

    
def docs2bow(sample_size=2000):
    file_path = os.path.join(params().paths['lda'], 'corpus.mm')
    mm = gensim.corpora.mmcorpus.MmCorpus(file_path)  # `mm` document stream now has random access
    if sample_size is not None:
        sample = [random.randint(0,mm.num_docs) for i in range(sample_size)]
        corpus_bow = []
        for doc in sample:
            corpus_bow.append(mm[doc])
    else:
        corpus_bow = []
        for doc in range(0,mm.num_docs,1):
            corpus_bow.append(mm[doc])
    word_ids = [item for sublist in corpus_bow for item in sublist]
    df = pd.DataFrame(word_ids, columns=['word','count'], dtype='int')
    df = df.groupby(['word'])['count'].sum().reset_index()
    bow = [tuple(x) for x in df.values]
    return bow

def corpus2bow(lda_instance):
    """Returns test corpus in bow format: list of (word_id,word_count)
    """
    lda_instance.dictionary[1]
    bow_dict = copy.deepcopy(lda_instance.dictionary.id2token)
    bow_dict = {k: 0 for (k,v) in bow_dict.items()}
    file_path = os.path.join(params().paths['lda'], 'corpus_test.mm')
    mm = gensim.corpora.mmcorpus.MmCorpus(file_path)  # `mm` document stream now has random access
    for doc in range(0,mm.num_docs,1):
        doc_dict = dict(mm[doc])
        for k, v in doc_dict.items():
            bow_dict[k] = bow_dict[k] + v
    bow_list = [(k, v) for k, v in bow_dict.items()] 
    return bow_list

def get_word_proba(bow,lda_instance):
    """Returns probability matrix of same format as lda_model.get_topics() for test corpus,
    corrects for missing probabilities due to missing words in test corpus by adding zero padding.
    """
    test_topics, test_word_topics, test_word_proba = lda_instance.lda_model.get_document_topics(bow, 
                                                                                                minimum_probability=0.0, 
                                                                                                minimum_phi_value=0.0, 
                                                                                                per_word_topics=True)
    
    placeholder = [(j,0.0) for j in range(lda_instance.lda_model.num_topics)]
    for i,t in enumerate(test_word_proba):
        if t[1] is None:
            test_word_proba.append((i,placeholder))
        elif not len(t[1]):
            #print(test_word_proba[i]) 
            test_word_proba[i] = (i,placeholder)
            test_word_proba.sort()
        elif len(t[1]) is not lda_instance.lda_model.num_topics:
            #print(test_word_proba[i]) 
            dict_ = dict(t[1])
            for j in range(lda_instance.lda_model.num_topics):
                try:
                   dict_[j]
                except KeyError:
                   dict_[j] = 0.0
            dict_list = list(dict_.items())
            dict_list.sort()
            test_word_proba[i] = (i,dict_list)
            test_word_proba.sort()

    test_word_proba = [[i[1] for i in g[1]] for g in test_word_proba]
    #transform to probabilities
    test_word_proba = np.transpose(np.apply_along_axis(lambda x: x/x.sum(),0,np.array(test_word_proba)))
    return test_topics, test_word_topics, test_word_proba

def jsd_measure(lda_instance):
    """
    returns a jensen-shannon distance measure between the empirical 
    word distribution in a holdout test corpus and the 
    distribution generated by the trained model.
    """
    bow = corpus2bow(lda_instance)
    test_topics, test_word_topics, test_word_proba = get_word_proba(bow,lda_instance)
    #set missing topics to zero
    if len(test_topics) is not lda_instance.lda_model.num_topics:
        topic_dict = dict(test_topics)
        for j in range(lda_instance.lda_model.num_topics):
            try:
               topic_dict[j]
            except KeyError:
               topic_dict[j] = 0.0
        test_topics = list(topic_dict.items())
    
    model_pdist = np.transpose([i[1] for i in test_topics]).dot(test_word_proba)
    bow_ = np.array([i[1] for i in bow])
    bow_ = bow_ / bow_.sum()
    bow_pdist = [(i[0][0],i[1]) for i in zip(bow,bow_)]
    model_pdist = [(i[0][0],i[1]) for i in zip(bow,model_pdist)]
    jsd = gensim.matutils.jensen_shannon(bow_pdist,model_pdist)
    return jsd
    
def get_perplexity(lda_model, lda_instance, chunksize=2000):
    file_path_test = os.path.join(params().paths['lda'], 'corpus_test.mm')
    mm = gensim.corpora.mmcorpus.MmCorpus(file_path_test)  # `mm` document stream now has random access
    sample = [random.randint(0,mm.num_docs) for i in range(chunksize)]
    test_corpus = []
    for doc in sample:
        test_corpus.append(mm[doc])
    perplexity = np.exp2(-lda_model.log_perplexity(test_corpus,len(lda_instance.articles)))
    return perplexity

def _get_scaled_significance(lda_model, n_words=200):
    _cols = ['token_id', 'weight', 'topic']
    df = pd.DataFrame(data=None, columns=_cols)
    num_topics = lda_model.num_topics
    for i in range(0,num_topics,1):
        _list = lda_model.get_topic_terms(i,n_words)
        df_n = pd.DataFrame(_list, columns=_cols[0:2])
        df_n['topic'] = i
        df = df.append(df_n)
        
    df = df.set_index(['token_id','topic'])
    df = df.join(df.groupby(level=0).sum(), how='inner', rsuffix='_sum').groupby(level=[0,1]).first()
    df['scaled_weight'] = df['weight']/df['weight_sum']
    return df['scaled_weight']

def get_unique_words(lda_instance, topn=10):
    """Builds df with topic words sorted by scaled uniqueness. 
    args:
        lda_instance (obj): Instance of LDA
        topn (int): top words to consider when sorting
    returns:
        sorted DataFrame
    """
    df_out = pd.DataFrame(data=None, columns=['scaled_weight','word'])

    df = _get_scaled_significance(lda_instance.lda_model, topn)
    for i in range(0,lda_instance.lda_model.num_topics,1):
        tokens = []
        df_topic = df[df.index.get_level_values('topic') == i]
        df_topic = df_topic[0:topn]
        for t in range(0,topn,1):
            tokens.append(lda_instance.dictionary[df_topic.index.values[t][0]])
        tokens = pd.DataFrame(tokens, index=df_topic.index, columns=['word'])
        df_topic = pd.concat([df_topic,tokens], axis=1, sort=True)

        df_out = df_out.append(df_topic)
    df_out.index = pd.MultiIndex.from_tuples(df_out.index)
    df_out = df_out.rename_axis(['token_id','topic'])
    df_out = df_out.sort_values(by = ['topic', 'scaled_weight'], ascending = [True, False])

    return df_out

def merge_documents_and_topics(lda_instance):
    """
    Merges topic weights to each article in the sample and outputs to HDF5.
    """
    # Load parsed articles
    print("Merging documents and LDA-topics")
    articles = read_h5py(os.path.join(params().paths['parsed_news'],
                                      params().filenames['parsed_news']))

    print("\tLoaded {} enriched documents ({} unique)... {}".format(len(articles),
                                                                    len(articles['article_id'].unique()),
                                                                    timestamp()))

    # Find LDA-document indices that match the ids of the enriched articles
    enriched_article_id = set(articles['article_id'])
    
    # Convert article_id to int
    lda_instance.article_id = [int(i) for i in lda_instance.article_id]
    
    lda_indices = [i for (i, j) in enumerate(lda_instance.article_id) if j in enriched_article_id]
    print("\t{} common article-ids... {}".format(len(lda_indices), timestamp()))

    # Find document-topics for the document-intersection above
    with Pool(6) as pool:
        document_topics = pool.map(partial(lda_instance.get_topics,
                                           lda_instance.lda_model,
                                           lda_instance.dictionary),
                                   [lda_instance.articles[i] for i in lda_indices])

    df_lda = pd.DataFrame({'article_id': [lda_instance.article_id[i] for i in lda_indices],
                           'topics': [[x[1] for x in document_topics[i]] for i in range(len(document_topics))]})

    # Merge the enriched data onto LDA-projections
    df_enriched_lda = pd.merge(df_lda, articles[['article_id', 'headline', 'date']],
                               how='inner',
                               on='article_id')

    print("\tJoin between LDA-topics and enriched documents gave {} documents... {}".format(len(df_enriched_lda),
                                                                                            timestamp()))

    # Save enriched documents with their topics
    folder_path = params().paths['doc_topics']
    topics_path = os.path.join(folder_path, params().filenames['lda_merge_doc_topics_file'])
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    df2 = pd.DataFrame(df_enriched_lda.topics.values.tolist(), index = df_enriched_lda.index)
    df_enriched_lda = pd.concat([df2, df_enriched_lda[['article_id', 'headline', 'date']]], axis=1)
    del(df2)
    df_enriched_lda.to_hdf(os.path.join(folder_path,topics_path), 'table', format='table', mode='w', append=False)

def merge_unseen_docs(lda_instance, start_date=None, end_date=None):
    articles = read_h5py(os.path.join(params().paths['parsed_news'],
                                      params().filenames['parsed_news']))

    #subset to date range
    if start_date is not None and end_date is not None:
        articles = articles.loc[(articles.date >= start_date) & (articles.date < end_date)]
        filename = params().filenames['lda_merge_doc_topics_file'] + '_' + start_date.strftime(
            "%Y-%m-%d") + '_' + end_date.strftime("%Y-%m-%d")
    else:
        filename = params().filenames['lda_merge_doc_topics_file']

    texts = list(zip(articles['article_id'].values, articles['body'].values))

    # Find document-topics for the document-intersection above
    with Pool(6) as pool:
        document_topics = pool.map(partial(lda_instance.get_topics,
                                           lda_instance.lda_model,
                                           lda_instance.dictionary),
                                   [i[1] for i in texts])

    df_lda = pd.DataFrame({'article_id': [i[0] for i in texts],
                           'topics': [[x[1] for x in document_topics[i]] for i in range(len(document_topics))]})

    df_lda = pd.merge(df_lda, articles[['article_id', 'headline', 'date']],
                               how='inner',
                               on='article_id')
    print(df_lda.columns.to_list())

    folder_path = params().paths['doc_topics']
    topics_path = os.path.join(folder_path, filename)

    df2 = pd.DataFrame(df_lda.topics.values.tolist(), index = df_lda.index)
    df_enriched_lda = pd.concat([df2, df_lda[['article_id', 'headline', 'date']]], axis=1)
    print(df_enriched_lda.columns.to_list())
    del(df2)
    df_enriched_lda.to_hdf(topics_path,
                           'table', format='table', mode='w', append=False)
            
def generate_wordclouds(lda_instance, topics=None, shade=True, title=None, num_words=15):
    """Generates word cloud images and saves to disk.
    args:
        topics: list of topics to cloud jointly, or int for single topics. 
            None (default) draws every topic in a separate image.
        shade: Set greyshade of word by "uniqueness" ranking. Less unique words are lighter.
        num_words: Number of words in cloud.
    """
    
    class MyColorFunctor():
      def __init__(self,df,cmap):
        self.df = df
        self.cmap = cmap
    
      def __call__(self,word,font_size,position,orientation,random_state=None,**kwargs):
        idx = int(self.df[self.df['word']==word]['index'])
        #convert cmap color at index to RGB integer format
        print(tuple([int(255*x) for x in cmap(idx)[:3]]))
        return tuple([int(255*x) for x in cmap(idx)[:3]])
    
    colors = ['#c1c1c2','#666666','#000000']
    cmap = LinearSegmentedColormap.from_list("mycmap", colors, N=num_words)

    print("Generating wordclouds... {}:".format(timestamp()))

    print("\t{} topics...".format(lda_instance.lda_model.num_topics))

    folder_path = os.path.join(params().paths['lda'], 'wordclouds_' + str(lda_instance.lda_model.num_topics))
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if topics:
        if isinstance(topics, list) and shade:
            print("Warning: Will not shade by uniqueness with multiple topics")
    
    topic_list = lda_instance.lda_model.show_topics(formatted=False, num_topics=-1, num_words=num_words)
    if shade:
        df = pd.DataFrame([i[1] for i in topic_list]).T
        dfu = get_unique_words(lda_instance, topn=num_words)

    if not topics:
        for t in range(0, lda_instance.lda_model.num_topics):
            if shade:
                words = pd.DataFrame(df[t].tolist(), index=df.index, columns=['word', 'weight'])        
                shades = dfu.xs(t, level=1, drop_level=False)
                words = words.merge(shades, on='word').sort_values(by='scaled_weight').reset_index(drop=True).reset_index()
                
                cloud = WordCloud(background_color='white', font_path='C:/Users/Erik/AppData/Local/Microsoft/Windows/Fonts/Nationalbank-Bold.ttf', stopwords=[],
                                  collocations=False, color_func=MyColorFunctor(words,cmap), max_words=200, width=1000, height=600)
                
            else:   
                cloud = WordCloud(background_color='white', font_path='C:/Users/Erik/AppData/Local/Microsoft/Windows/Fonts/Nationalbank-Bold.ttf', stopwords=[],
                                  collocations=False, colormap=cmap, max_words=200, width=1000, height=600)
            
            topic_words = dict(topic_list[t][1])
            cloud.generate_from_frequencies(topic_words)                
            plt.gca().imshow(cloud)
            #plt.gca().set_title('Topic {}'.format(t), fontdict=dict(size=12, fontname='Nationalbank'))
            plt.gca().axis('off')
    
            file_path = os.path.join(folder_path, '{}.png'.format(t))
            plt.savefig(file_path, bbox_inches='tight', dpi=300)
    else:
        word_dict = lda_instance.dictionary.id2token
        num_tokens = len(lda_instance.dictionary)
        if isinstance(topics, int):
            if shade:
                words = pd.DataFrame(df[topics].tolist(), index=df.index, columns=['word', 'weight'])        
                shades = dfu.xs(topics, level=1, drop_level=False)
                words = words.merge(shades, on='word').sort_values(by='scaled_weight').reset_index(drop=True).reset_index()
                print(words)
                cloud = WordCloud(background_color='white', font_path='C:/WINDOWS/FONTS/Nationalbank-Bold.TTF', stopwords=[],
                                  collocations=False, color_func=MyColorFunctor(words,cmap), max_words=200, width=1000, height=600)
            else:
                cloud = WordCloud(background_color='white', font_path='C:/WINDOWS/FONTS/Nationalbank-Bold.TTF', stopwords=[],
                                      collocations=False, color_func=lambda *args, **kwargs: "black", max_words=200, width=1000, height=600)
                
            topic_words = dict(topic_list[topics][1])  
            if not title:
                file_path = os.path.join(folder_path, f'{title}.png')
                title = f'Topic {topics}'
            else:
                file_path = os.path.join(folder_path, f'{title}.png')
        else:
            cloud = WordCloud(background_color='white', font_path='C:/WINDOWS/FONTS/Nationalbank-Bold.TTF', stopwords=[],
                              collocations=False, color_func=lambda *args, **kwargs: "black", max_words=200, width=1000, height=600)
            file_path = os.path.join(folder_path, '{}.png'.format('-'.join(str(x) for x in topics)))
            topic_list = [t for (i,t) in enumerate(topic_list) if i in topics]
            dft = pd.DataFrame(range(num_tokens), columns=['token_id'])
            for i,t in enumerate(topics):
                #topic_words = dict(topic_list[t][1])
                df = pd.DataFrame(lda_instance.lda_model.get_topic_terms(t,num_tokens),
                                  columns=['token_id',f'weight{t}'])
                dft = dft.merge(df, on='token_id')
            dft = pd.DataFrame(dft.loc[:, [x for x in dft.columns if x.startswith('weight')]].mean(axis=1), columns=['weight']).reset_index()
            dfw = pd.DataFrame(word_dict.values(), index=word_dict.keys(), columns=['word']).reset_index()
            dft = dft.merge(dfw, on='index', how='inner')
            dft = dft.sort_values('weight',ascending=False).iloc[0:15,:]
            
            topic_words = dict(list(zip(*map(dft.get, ['word','weight']))))
            if not title:
                title = 'Topic(s): {}'.format(', '.join(str(x) for x in topics))
        
        cloud.generate_from_frequencies(topic_words)
        plt.gca().imshow(cloud)
        plt.gca().set_title(title, fontdict=dict(size=12), fontname='Nationalbank')
        plt.gca().axis('off')
        plt.savefig(file_path, bbox_inches='tight', dpi=300)
        
def term_frequency(corpus_bow, dictionary, terms=30):
    corpus_iter = iter(corpus_bow)
    counter = Counter()
    while True:
        try:
            document = next(corpus_iter)
            for token_id, document_count in document:
                counter[dictionary.get(token_id)] += document_count
        except StopIteration:
            print("Done counting term frequencies")
            break
    return counter.most_common(terms), counter.most_common()[:-terms-1:-1]

def _return_array(array, n_topics=80):
    """
    Utility function transforming the projections as returned by the LDA
    module into a 1xT numpy array, where T is the number of topics.
    """
    output = np.array(
        [[sum([el[1] for el in row if el[0] == topic]) for topic in range(1, n_topics + 1)] for row in array])
    return output

def parse_topic_labels(name,num_topics):
    """
    reads hand labeled topics from json file.
    
    """
    label_path = os.path.join(params().paths['topic_labels'], 
                              name+str(num_topics)+'.json')
      
    with codecs.open(label_path, 'r', encoding='utf-8-sig') as f:
        labels = json.load(f)
    return labels
