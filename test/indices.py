import os
import pickle
import json
import re
import copy
import h5py
import codecs
import numpy as np
import pandas as pd
import matplotlib.dates as mdates

from scipy.signal import detrend
from gensim.models import KeyedVectors
from cycler import cycler
from datetime import datetime
from nltk.stem.snowball import SnowballStemmer
from multiprocessing import Pool
from functools import partial
from matplotlib import pyplot as plt
from src.fui.utils import params, read_h5py, dump_csv
from sklearn.preprocessing import StandardScaler

class BaseIndexer():
    """Base class for constructing indicies.
    init_args:
    name (str): Name of the index. Ex: "international_trade_u_weighted"
    """
    def __init__(self,name):
        self.name = name
        
    def validate(self):
        """Method for calculating correlations with macro time-series 
        and index/vix/vdax
        """
        v1x, vix = self.load_vix(f='M')
        data = pd.read_csv(params().paths['input']+'validation.csv', header=0)
        data['date'] = pd.to_datetime(data['date'], format="%Ym%m") + pd.tseries.offsets.MonthEnd(1)
        data = data.set_index('date')

        idx = self.idx.filter(regex='_norm', axis=1)
        
        for var in data.columns:
            data[var] = _normalize(data[var])
            self.corr[var] = \
                (_calc_corr(idx, data[var]),
                 _calc_corr(v1x, data[var]),
                 _calc_corr(vix, data[var]))
        return self.corr
    
    def load_vix(self,frq='M'):
        """Loads vix and vdax from csv files
        args:
        frq = observation frequency for vix
        returns: (df,df)
        """
        # v1x = pd.read_csv(params().paths['input']+'v1x_monthly.csv',
        #                   names=['date','v1x'], header=0)
        #
        # v1x['date'] = pd.to_datetime(v1x['date'])
        # v1x.set_index('date', inplace=True)
        # v1x = v1x[str(self.start_year):str(self.end_year)+self.end_str]
        # v1x['v1x'] = _normalize(v1x['v1x'])

        vix = pd.read_csv(params().paths['input'] + 'vixcurrent.csv',
                          names=['date', 'vix'], header=1)
        vix['date'] = pd.to_datetime(vix['date'])
        vix.set_index('date', inplace=True)
        vix = vix.resample(frq).mean()
        vix.columns = vix.columns.get_level_values(0)
        vix = vix[str(self.start_year):str(self.end_year)+self.end_str]
        vix['vix'] = _normalize(vix['vix'])
                   
        return vix
    
    def parse_topic_labels(self, name):
        """
        reads hand labeled topics from json file.
        args:
        name (str): name of json file (num_topics must be suffix)
        returns: 
        (dict) with labels
        """
        label_path = os.path.join(params().paths['topic_labels'], 
                                  name+str(self.num_topics)+'.json')
          
        with codecs.open(label_path, 'r', encoding='utf-8-sig') as f:
            self.labels = json.load(f)
        self.labels['EP'] = list(set().union(self.labels['EP_int'], self.labels['EP_dk']))
        return self.labels
          

    def aggregate(self, df, col='idx', norm=True, write_csv=True, method='mean', lda=True):
        """
        Aggregates to means within 
        each aggregation frequency
        args:
        df (DataFrame): input data
        col (str): column to aggregate
        norm (bool): add column of normalized values
        write_csv (bool): write result as csv.
        method (str): agg_func, 'mean' or 'sum'
        returns:
        DataFrame of aggregation result with datetime index.
        """     
        cols = [col, 'date']
        if lda:
            cols.extend(self.topics)

        df.set_index('date', inplace=True, drop=False)
        idx = df[cols].groupby(
            [pd.Grouper(key='date', freq=self.frq)]
        ).agg([method])

        if norm:
            scaler = StandardScaler()
            idx = pd.DataFrame(scaler.fit_transform(idx), columns=idx.columns, index=idx.index)

        idx = idx[str(self.start_year):str(self.end_year)+self.end_str]
        idx.columns = idx.columns.get_level_values(0)
        print("Last month: ", idx[-1:])
        #idx.to_pickle(params().paths['indices']+self.name+'_'+self.frq+'.pkl')
        if write_csv:
            dump_csv(params().paths['indices'], self.name+'_'+self.frq, idx.iloc[:,0], verbose=False)
        return idx

    def plot_index(self, plot_vix=False, plot_bloom=False, annotate=True, title=None):
        """
        Plot index from df column named "idx_norm".
        Args:
        plot_vix (bool): Add series of vix and vdax to plot.
        plot_hh (bool): Plot household equity transactions (not ready).
        annotate (bool): Add event annotation to plot.
        title (str): Plot title.
        Returns: plt objects (figure,axes).
        """

        years = mdates.YearLocator()  # every year
        months = mdates.MonthLocator((1, 4, 7, 10))
        years_fmt = mdates.DateFormatter('%Y')

        out_path = params().paths['indices']
        c = cycler(
            'color',
            [
                (0/255, 123/255, 209/255),
                (146/255, 34/255, 156/255),
                (196/255, 61/255, 33/255),
                (223/255, 147/255, 55/255),
                (176/255, 210/255, 71/255),
                (102/255, 102/255, 102/255)
            ])
        plt.rcParams["axes.prop_cycle"] = c

        fig, ax = plt.subplots(figsize=(14,8))
        ax.plot(self.idx.index, self.idx['idx'], label='Børsen Uncertainty Index')
        if title:
            ax.title.set_text(title)
        if plot_vix:
            vix = self.load_vix(self.frq)
            #ax.plot(v1x.index, v1x.v1x, label='VDAX-NEW')
            ax.plot(vix.index, vix.vix, label='VIX')
        if plot_bloom:
            bloom = pd.read_csv(params().paths['indices'] + 'bloom_'+self.frq+'.csv',
                              names=['date', 'bloom'], header=0)
            bloom['date'] = pd.to_datetime(bloom['date'])
            bloom.set_index('date', inplace=True)
            ax.plot(bloom.index, bloom.bloom, label='BBD')


        ax.legend(frameon=False, loc='upper left')    
    
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)
        ax.xaxis.set_minor_locator(months)

        if annotate:
            ax.axvspan(xmin=datetime(2000,1,31), xmax=datetime(2000,5,31), 
                       color=(102/255, 102/255, 102/255), alpha=0.3)
            ax.annotate("Dot-com \n crash", xy=(datetime(2000,3,31), 0.8),
                        xycoords=('data', 'axes fraction'), fontsize='large', ha='center')
            ax.axvspan(xmin=datetime(2011,3,1), xmax=datetime(2012,11,30), 
                       color=(102/255, 102/255, 102/255), alpha=0.3)
            ax.annotate("Debt crisis", xy=(datetime(2012,2,15), 0.96),
                        xycoords=('data', 'axes fraction'), fontsize='large', ha='center')
            #ax.axvspan(xmin=datetime(2018,3,1), xmax=datetime(2019,12,1),
            #           color=(102/255, 102/255, 102/255), alpha=0.3)
            #ax.annotate("Trade war", xy=(datetime(2019,2,15), 0.97),
            #            xycoords=('data', 'axes fraction'), fontsize='large', ha='center')
            ax.axvspan(xmin=datetime(2020,2,1), xmax=datetime(2020,7,1),
                       color=(102/255, 102/255, 102/255), alpha=0.3)
            ax.annotate("COVID-19", xy=(datetime(2020,3,15), 0.96),
                        xycoords=('data', 'axes fraction'), fontsize='large', ha='center')

            dates_dict = {'Euro \nreferendum': ('2000-09-28', 0.5),
                          '9/11': ('2001-09-11', 0.7),
                          '2001\n election': ('2001-11-20', 0.8),
                          'Invasion of Iraq': ('2003-03-19', 0.9),
                          '2005\nelection': ('2005-02-08', 0.8),
                          'Northern Rock\n bank run': ('2007-09-14', 0.9),
                          '2007\n election': ('2007-11-13', 0.8),
                          'Lehman Brothers': ('2008-09-15', 0.97),
                          '2010 Flash Crash': ('2010-05-06', 0.9),
                           '2011 election': ('2011-09-15', 0.8),
                          '"Whatever\n it takes"': ('2012-07-26', 0.7),
                          '2013 US gov\n shutdown': ('2013-10-15', 0.9),
                           #"'DKK pressure\n crisis': ('2015-02-15', 0.7),
                          '2015\n election': ('2015-06-18', 0.9),
                          'Migrant\n crisis': ('2015-09-15', 0.8),
                          'Brexit': ('2016-06-23', 0.75),
                          'US\n election': ('2016-11-08', 0.9),
                           #'Labor parties\n agreement': ('2018-04-15', 0.7),
                          'Danske Bank\n money laundering': ('2018-09-15', 0.9),
                          '2018 US gov\n shutdown': ('2018-12-10', 0.8)}

            for l, d in zip(dates_dict.keys(), dates_dict.values()):
                date = datetime.strptime(d[0], "%Y-%m-%d")
                ax.axvline(x=date, color=(102 / 255, 102 / 255, 102 / 255), alpha=0.3, linewidth=2)
                ax.annotate(l, xy=(date, d[1]), xycoords=('data', 'axes fraction'),
                            fontsize='medium', ha='center')
            #corr = _calc_corr(vix,idx[idx_name])
            #ax.text(0.80, 0.95, 'Correlation with VIX: %.2f' % round(corr,2) , transform=ax.transAxes)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_ylabel("Standard deviations", fontsize='large')
        for label in ax.xaxis.get_ticklabels()[::2]:
            label.set_visible(False)

        plt.tight_layout()
        plt.savefig(f'{out_path}{self.name}_{self.frq}_plot.png', dpi=300)
        return fig, ax
    
class LDAIndexer(BaseIndexer):
    """Class for constructing LDA indices."""    
    
    def build(self, df=None, labels='meta_topics', num_topics=80, start_year=2000, end_year=2020,
              frq='Q', topics=None, xsection=None, topic_thold=0.0, norm=True,
              xsection_thold=0.1, main_topic=False, extend_u_dict=True, u_weight=True):
        """Calculates indicies.
        
        args:
        df (DataFrame): input data, if None then merge_lda_u().
        labels (str): name of label json dict to use.
        num_topics (int): number of topics in the LDA model.
        start_year (int), end_year (int)=2019: window for the index.
        frq (str): Final aggregation frequency, 'D', 'M' or 'Q'.
        topics (optional, str, int or list of int): Topics to use, can be specifc as 
            - topic_ids (int or list of ints)
            - name of label dict
        xsection (optional, list of str): list of label dicts with sets of topics 
        topic_thold: probability threshold for a topics to be considered.
        xsection_thold: probability threshold for topics in xsection to be considered.
        main_topic (bool): Only consider highest probability topic. 
        Must provide `topics´ if True.
        extend_u_dict (bool): Use extended set of u-words.
        u_weight (bool): Weight index by share of u-words.
        
        returns:
        DataFrame of aggregated index. Also accessible as attribute `idx´.
        """

        self.start_year = start_year
        self.end_year = end_year
        self.frq = frq
        self.num_topics = num_topics
        if labels:
            self.label_dict = self.parse_topic_labels(labels)
        if end_year == 2020:
            self.end_str = '-06-30'
        else:
            self.end_str = ''

        assert (topics is not None) or (xsection is not None), "Must provide a list or dict of topics."
        assert topic_thold >= 0.0 and xsection_thold >= 0.0, "No negative thresholds."
        

        if df is None: 
            df = merge_lda_u(extend_u_dict, num_topics=self.num_topics)
        
        if main_topic:
            df['idx'] = df['topics'].apply(
                lambda x : bool(np.argmax(x) in set(topics))*1)
            if u_weight:
                df['idx'] = df['idx']*(df['u_count']/df['word_count'])
            self.idx = self.aggregate(df)
            return self.idx

        if xsection:
            assert isinstance(xsection, dict) or isinstance(xsection, list), \
            "intersection_dict must be dict or list of dict keys"
            if isinstance(xsection, list):
                xsection = dict((k, self.label_dict[k]) for k in xsection)
                
            cat = []
            for k, t in xsection.items():
                if isinstance(t, int):
                    t = [t]
                assert isinstance(t, list) , "Values in intersection dict must be int or list."
                df[str(k)] = df['topics'].apply(
                        lambda x : np.array([x[i] for i in t if x[i] >= topic_thold]).sum())
                cat.append(str(k))
            df['idx'] = df.loc[:, cat].prod(axis=1)*\
                        (df[cat] > xsection_thold).all(1).astype(int) 
            if u_weight:
                df['idx'] = df['idx']*((df['n_count']+df['u_count'])/df['word_count'])

            self.idx = self.aggregate(df, norm=norm)
            return self.idx

        if isinstance(topics, int):
            topics = [topics]
        assert isinstance(topics, list), "Keyword topics must be list or int."
        if isinstance(topics[0], str):
            topics = self.label_dict[topics[0]]
        self.topics = topics
        print(self.topics)
        df['idx'] = df['topics'].apply(
                lambda x : np.array([j for i, j in enumerate(x) if (i in topics)]).sum())
        df.loc[df.idx < topic_thold, 'idx'] = 0
        df['idx_components'] = df['topics'].apply(
                lambda x : [j for i, j in enumerate(x) if (i in topics)])
        df_comp = pd.DataFrame(df.idx_components.values.tolist(), index=df.index)
        df_comp.loc[df.idx == 0] = 0
        df_comp.rename(columns={i: j for i, j in enumerate(topics)}, inplace=True)
        df = pd.concat([df[['date', 'idx', 'u_count', 'word_count']], df_comp], axis=1)
        if u_weight:
            df['idx'] = df['idx'].mul(df['u_count'], axis=0).div(df['word_count'], axis=0)
            #contribution of each topic is normalized to sum to one (last div operation below)
            df[df.columns[-len(topics):]] = df[df.columns[-len(topics):]].mul(df['u_count'], axis=0)\
                .div(df['word_count'], axis=0)
                #.div(df['idx'], axis=0)
            df[df.columns[-len(topics):]].fillna(0, inplace=True)
        else:
            df[df.columns[-len(topics):]] = df[df.columns[-len(topics):]].div(df['idx'], axis=0)
        #return df

        self.idx = self.aggregate(df, norm=norm)
        return self.idx

class BloomIndexer(BaseIndexer):
    def build(self, logic, bloom_dict_name,
              start_year=2000, end_year=2020, frq='Q', u_weight=False, extend=True):
        """
        Finds for articles containing words in bloom dictionary. Saves result to disk.
        args:
        dict_name: name of bloom dict in params
        logic: matching criteria in params
        """
        self.start_year = start_year
        self.end_year = end_year
        self.logic = logic
        self.frq = frq
        if self.end_year == 2020:
            self.end_str = '-06-30'
        else:
            self.end_str = ''
        
        out_path = params().paths['indices']+self.name+'\\'+self.logic
        if not os.path.exists(out_path):
            os.makedirs(out_path)        

        if extend:
            bloom_dict = extend_dict_w2v(bloom_dict_name, n_words=10)
            df = read_h5py(os.path.join(params().paths['enriched_news'],
                                       params().filenames['parsed_news_uc_ext']))
        else:
            bloom_dict = params().dicts[bloom_dict_name]
            df = read_h5py(os.path.join(params().paths['parsed_news'],
                                        params().filenames['parsed_news']))

        b_E, b_P, b_U = _get_bloom_sets(bloom_dict)
        print('\n\nEconomic words: ' + repr(b_E) +
              '\n\n Political words: ' + repr(b_P) +
              '\n\n Uncertainty words: ' + repr(b_U))
        
        #stem articles
        with Pool() as pool:
            df['body_stemmed'] = pool.map(_stemtext, 
                                          df['body'].values.tolist())
        if u_weight:
            logic_str = params().options['bloom_logic_weighted']
        else:
            logic_str = params().options['bloom_logic'][self.logic]
            
        print('\nLogic: '+logic_str)
        #compare to dictionary
        with Pool() as pool:
            df['idx'] = pool.map(partial(_bloom_compare, 
                                         logic=logic_str, 
                                         bloom_E=b_E, 
                                         bloom_P=b_P, 
                                         bloom_U=b_U), 
                                         df['body_stemmed'].values.tolist())
        if u_weight:
            df['idx'] = df['idx']*((df['n_count']+df['u_count'])/df['word_count'])

        self.idx = self.aggregate(df, norm=True, lda=False)
        return self.idx

def _calc_corr(df1,df2):
    df1 = df1.join(df2, how='inner', on='date')
    corr_mat = pd.np.corrcoef(df1.iloc[:,0].tolist(), df1.iloc[:,1].tolist())
    return corr_mat[0,1]

def _bloom_compare(word_list, logic, bloom_E, bloom_P, bloom_U):  
    stem_set = set(word_list)
    return eval(logic)

def _get_bloom_sets(bloom_dict):
    b_E = set(bloom_dict['economic'])
    b_P = set(bloom_dict['political'])
    b_U = set(bloom_dict['uncertainty'])
    return b_E, b_P, b_U

def _normalize(series):
    return (series-series.mean())/series.std()

def _count(word_list, word_set):
    count = 0
    for word in word_list:
        if word in word_set:
            count += 1
    return count

def _count_n(text, word_list):
    count = 0
    for word in word_list:
        count += text.count(word)
    return count


def _check_stem_duplicates(word_list):
    """
    Stems list of words and removes any resulting duplicates
    """
    stemmer = SnowballStemmer("danish")
    stemmed_list = [stemmer.stem(word) for word in word_list]
    #remove duplicates after stemming
    stemmed_list = list(dict.fromkeys(stemmed_list))
    return stemmed_list

def _stemtext(text, min_len=2, max_len=25):
    # Remove any non-alphabetic character, split by space
    stemmer = SnowballStemmer("danish")
    pat = re.compile(r'(((?![\d])\w)+)', re.UNICODE)

    text = text.lower()
    list_to_stem = []
    list_to_stem = [match.group() for match in pat.finditer(text)]
    
    stemmed_list = [stemmer.stem(word) for word in list_to_stem if len(word) >= min_len and len(word) <= max_len]
    return stemmed_list

def _load_u_count(sample_size=0,extend=True):
    if extend:
        filename = params().filenames['parsed_news_uc_ext']
    else:
        filename = params().filenames['parsed_news_uc']

    file_path = os.path.join(params().paths['enriched_news'],filename)
    df = read_h5py(file_path)
    df = df[['article_id', 'body', 'u_count', 'n_count', 'word_count']]
    if sample_size > 0:
        return df.sample(sample_size)
    else:
        return df

def merge_lda_u(extend=True,sample_size=0,num_topics=90):
    """Merges uncertainty counts and topic vectors.
    args:
        extend (bool): Use extended set of u-words.
        sample_size (int): Return a random sample of articles.
        num_topics (int): LDA model to use.
    returns:
        DataFrame with columns 'topics' and 'u_count'
    """
    if extend:
        suffix='u_count_extend'
    else:
        suffix='u_count'
    try:
        df = pd.read_hdf(params().paths['doc_topics']+'doc_topics_'+suffix+'.h5', 'table')
        df['date'] = pd.to_datetime(df['date'])
        
        #convert columns to single col list
        df['topics']= df.iloc[:,0:num_topics].values.tolist()
        df.drop(df.columns[0:num_topics], axis=1, inplace=True)
        if sample_size > 0:
            return df.sample(sample_size) 
        return df
    
    except FileNotFoundError:
        print('File not found, merging lda topics and uncertainty counts...')
        df_u = _load_u_count(extend=extend, sample_size=sample_size)
        df = pd.read_hdf(params().paths['doc_topics']+'document_topics.h5', 'table')
        df['topics']= df.iloc[:,0:num_topics].values.tolist()
        df.drop(df.columns[0:num_topics], axis=1, inplace=True)
        df = df.merge(df_u, 'inner', 'article_id')
        df['date'] = pd.to_datetime(df['date'])
        save_topics_to_hdf(df,suffix)

        if sample_size > 0:
            return df.sample(sample_size) 
        return df
    
def save_topics_to_hdf(df,suffix):
    df2 = pd.DataFrame(df.topics.values.tolist(), index = df.index)
    print(df.dtypes)
    df = pd.concat([df2, df[['article_id', 'date', 'u_count', 'n_count', 'word_count']]], axis=1)
    df['u_count'] = df['u_count'].astype(np.int64)
    df['n_count'] = df['n_count'].astype(np.int64)
    print(df.dtypes)
    del(df2)
    df.to_hdf(params().paths['doc_topics']+'doc_topics_'+suffix+'.h5', 'table', format='table', mode='w', append=False)

def extend_dict_w2v(dict_name, n_words=10):
    """
    Extends bloom dictionary with similar words using a pre-trained
    embedding. Default model: https://fasttext.cc/docs/en/crawl-vectors.html
    args:
    params: input_params.json
    dict_name: name of Bloom dict in params
    n_words: include n_nearest words to subject word.
    """
    model = KeyedVectors.load_word2vec_format(params().paths['w2v_model'], binary=False)
    print("Word2vec model loaded")
    dict_out = copy.deepcopy(params().dicts[dict_name])
    for k, v in params().dicts[dict_name].items():
        for val in v:
            #print('\n'+v)
            try:
                similar_words = [w[0] for w in model.most_similar(positive=val, topn=n_words)]
                dict_out[k].extend(_check_stem_duplicates(similar_words))
                #print('\n',model.most_similar(positive=v))
            except KeyError:
                continue
    return dict_out

def uncertainty_count(extend=True, workers=16):
    """
    Counts u-words in articles. Saves result as HDF to disk.
    args:
        extend (bool): Use extended set of u-words
    """
    if extend:
        U_set = set(list(params().dicts['uncertainty_ext'].values())[0])
        filename = params().filenames['parsed_news_uc_ext']
    else:
        U_set = set(list(params().dicts['uncertainty'].values())[0])
        filename = params().filenames['parsed_news_uc']

    print(U_set)
    #get parsed articles
    df = read_h5py(os.path.join(params().paths['parsed_news'],
                               params().filenames['parsed_news']))

    #stem articles
    with Pool(workers) as pool:
        df['body_stemmed'] = pool.map(_stemtext,
                                      df['body'].values.tolist())
    
    #compare to dictionary
    with Pool(workers) as pool:
        df['u_count'] = pool.map(partial(_count, 
                                 word_set=U_set), 
                                 df['body_stemmed'].values.tolist())
    
    
    N_list = list(params().dicts['negations'].values())[0]
    with Pool(workers) as pool:
        df['n_count'] = pool.map(partial(_count_n, 
                             word_list=N_list), 
                             df['body'])
        
    #save to disk
    df.drop(columns='body_stemmed', inplace=True)
    outpath = os.path.join(params().paths['enriched_news'],filename)
    with h5py.File(outpath, 'w') as hf:
        string_dt = h5py.string_dtype(encoding='utf-8')
        hf.create_dataset('parsed_strings', data=df, dtype=string_dt)