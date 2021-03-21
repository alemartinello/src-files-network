# -*- coding: utf-8 -*-
import os
import pandas as pd
import pickle
import h5py
import glob
import html
import numpy as np
from src.fui.utils import params

def parse_for_lda(nrows=None):
    """
    Loads the data from CSV and performs some basic cleaning. Essentially the
    cleaning removes corrupted lines.
    """
    # Load the data
    df, _ = import_csv()

    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['year'] = df['date'].dt.year
    df = df[df['year'] > 1999]

    #drop irrelevant section names
    start_n = df.shape[0] 
    df = df[~df['section_name'].isin(['Biler', 'Bolig-besøg', 'Bolig-indret','Bolig-items','Cannes','Cannes Lions',
                                     'Design', 'Digital Gadget', 'Gastronomi', 'Karriere', 'Kriminal', 'Kultur',
                                     'Livsstil', 'Magasin Pleasure', 'Magasin Pleasure 2. sektion', 'Magasin Pleasure 2. sektion Rejser',
                                     'Magasin Pleasure 2. sektion Rejser Hoteller Stil', 'Magasin Pleasure Biler',
                                     'Magasin Pleasure Design', 'Magasin Pleasure EM', 'Magasin Pleasure Firmabilen 2015',
                                     'Magasin Pleasure Interiør', 'Magasin Pleasure kunst & kultur', 'Magasin Pleasure Portræt',
                                     'Magasin Pleasure rejser', 'Magasin Pleasure Ure', 'Michelin', 'Motion', 'Play 2016',
                                     'Pleasure', 'Portræt', 'Profil & Karriere', 'Underholdning', 'Week-div', 'Week-golf',
                                     'Week-livsstil', 'Week-mad', 'Week-maritim', 'Week-mode', 'Week-motor', 'Week-rejser',
                                     'Weekend Diverse', 'Weekend Golf','Weekend Kultur','Weekend Livsstil',
                                     'Weekend Livstil','Weekend Mad','Weekend Maritim','Weekend Mode','Weekend Motor',
                                     'Weekend Outdoor', 'Weekend Rejser'])]
    end_n = df.shape[0]
    print('Dropped {} articles with irrelevant section names'.format(start_n-end_n))
    print(f'Current number of articles: {end_n}')

    #drop word count below 50
    df['word_count'] = df['body'].str.count(' ') + 1
    start_n = df.shape[0]
    df = df[df.word_count >= 50]
    end_n = df.shape[0]
    print('Dropped {} articles with less than 50 words'.format(start_n-end_n))
    print(f'Current number of articles: {end_n}')

    df['body'] = __clean_text(df['body'])

    # create unique row index
    df['article_id'] = df.reset_index().index
    print('Columns: ', df.columns)
    
    with h5py.File(os.path.join(params().paths['parsed_news'],params().filenames['parsed_news']), 'w') as hf:
        string_dt = h5py.string_dtype(encoding='utf-8')
        hf.create_dataset('parsed_strings', data=df, dtype=string_dt)

def __clean_text(series):
    """
    ProcessTools: Clean raw text
    """
    # Step 1: Remove html tags
    print('Step 1: Removing tags')
    series = series.str.replace(r'<[^>]*>','',regex=True)
    series = series.str.replace(r'https?:\/\/\S*','',regex=True)
    series = series.str.replace(r'www.\S*','',regex=True)
        
    # Step 2: Remove everything following these patterns (bylines)     
    print('Step 2: Removing bylines')
    pattern = "|".join(
        [r'\/ritzau/AFP', r'Nyhedsbureauet Direkt', r'\w*\@borsen.dk\b', r'\/ritzau/', r'\/ritzau/FINANS'])
    series = series.str.replace(pattern,'',regex=True)

    # Step 3: Remove \n, \t
    print('Step 3: Removing other')
    series = series.str.replace(r'\n', ' ')
    series = series.str.replace(r'\t', ' ')

    # Step 4: Remove additional whitespaces
    #series = series.str.split().str.join(' ')

    # Step 5: Convert to lowercase
    series = series.str.lower()
    
    # Step 6: Unescape any html entities
    print('Step 6: Unescape html')
    series = series.apply(html.unescape)
    
    # Manually remove some html
    series = series.str.replace(r'&rsquo', '')
    series = series.str.replace(r'&ldquo', '')
    series = series.str.replace(r'&rdquo', '')
    series = series.str.replace(r'&ndash', '')
    
    return series

def load_parsed_data(sample_size=None):
    filelist = glob.glob(params().paths['parsed_news']+'boersen*.pkl') 
    df = pd.DataFrame()
    for f in filelist:    
        with open(f, 'rb') as f_in:
            df_n = pickle.load(f_in)
            df = df.append(df_n)
    if sample_size is not None:
        return df.sample(sample_size)
    else:
        return df

def import_scraped_articles():
    path = params().paths['scraped']  # use your path
    all_files = glob.glob(path + "/*/scraped*.csv")

    dfs = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0, sep=";")
        dfs.append(df)
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df['date'] = df['date'].replace('404', np.nan)
    df = df.dropna(subset=['date'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates(subset=['headline_web', 'date'])
    df = df.drop(columns=['Unnamed: 0', 'Unnamed: 1', 'sep=','headline_q'])
    # url ledelse/ is missing proper date (date is date of scrape), drop these
    df = df.loc[df['url'].str.find('/ledelse/') == -1]
    df.rename(columns={'headline_web': 'headline', 'bodytext': 'body', 'url': 'byline_alt'}, inplace=True)
    return df

def import_csv():
    #Step 1: merge df1 and df2 on ID. Replace Eavis arkiv articles in df1 with merged counterpart.
    csvpath = \
        params().paths['boersen_articles']+params().filenames['boersen_csv']
    df1 = pd.read_csv(csvpath, sep=';', encoding='UTF-16', error_bad_lines=False)
    df1['ArticleDateCreated'] = pd.to_datetime(df1['ArticleDateCreated'], format="%Y-%m-%d", errors='coerce')
    df1 = df1.dropna(axis=0, subset=['ArticleDateCreated','ArticleContents', 'Title', 'ID'])
    df1['ID'] = pd.to_numeric(df1['ID'], errors='coerce')

    csvpath = \
        params().paths['boersen_articles']+params().filenames['boersen_csv2']
    df2 = pd.read_csv(csvpath, sep=';', encoding='UTF-16', error_bad_lines=False)
    df2['dateRelease'] = pd.to_datetime(df2['dateRelease'], format="%Y-%m-%d", errors='coerce')
    df2 = df2.dropna(axis=0, subset=['dateRelease', 'headline', 'id', 'content'])
    df2['id'] = pd.to_numeric(df2['id'], errors='coerce')

    df1 = df1.merge(df2, left_on='ID', right_on='id', how='left')
    df1['ArticleContents'].loc[df1['Supplier'] =='E-Avis arkiv'] = df1['content']
    df1['Title'].loc[df1['Supplier'] =='E-Avis arkiv'] = df1['headline']
    df1['Author'].loc[df1['Supplier'] =='E-Avis arkiv'] = df1['byline']

    del df2

    keepcols = ['ID', 'Title', 'ArticleContents',
                'ArticleDateCreated',
                'Author', 'SectionName',
                'SAXoCategory', 'SAXoByline']

    df1 = df1[keepcols]

    df1.rename(columns={'ID':'id', 'Title':'headline', 'ArticleContents':'body',
                'ArticleDateCreated':'date',
                'Author':'byline', 'SectionName':'section_name',
                'SAXoCategory':'category', 'SAXoByline':'byline_alt'}, inplace=True)

    #Step 2: Keep articles until April 1st 2019. Use data from df3 for period after that.
    df1 = df1.loc[(df1['date'] < pd.Timestamp(2019,4,1))]

    csvpath = \
        params().paths['boersen_articles']+params().filenames['boersen_csv3']
    df3 = pd.read_csv(csvpath, sep=';', encoding='latin-1', error_bad_lines=False)
    df3['dateRelease'] = pd.to_datetime(df3['dateRelease'], format="%Y-%m-%d", errors='coerce')
    df3['id'] = pd.to_numeric(df3['id'], errors='coerce')

    keepcols = ['id', 'headline', 'content', 'dateRelease', 'byline',
                'saxoStrSection', 'saxoCategory',
                'saxoAuthor']

    df3 = df3[keepcols]

    df3.rename(columns={'id':'id', 'headline':'headline', 'content':'body',
                'dateRelease':'date',
                'byline':'byline', 'saxoStrSection':'section_name',
                'saxoCategory':'category', 'saxoAuthor':'byline_alt'}, inplace=True)

    df3 = df3.loc[(df3['date'] >= pd.Timestamp(2019,4,1))]
    df3 = df3.loc[(df3['date'] < pd.Timestamp(2020,3,1))]

    start_n = df3.shape[0]
    df3 = df3.dropna(axis=0, subset=['headline', 'id', 'body'])
    end_n = df3.shape[0]
    print('Dropped {} articles with NaN headline, id or content'.format(start_n-end_n))

    start_n = df3.shape[0]
    df3 = df3.drop_duplicates(subset=['headline', 'body'])
    end_n = df3.shape[0]
    print('Dropped {} articles with duplicate headline, id or content'.format(start_n-end_n))

    df1 = df1.append(df3)

    df4 = import_scraped_articles()
    df1 = df1.append(df4)
    df1 = df1.sort_values('date')

    del df3, df4

    df1['body'] = df1['body'].str.replace(r'<[^>]*>', '', regex=True)
    df1['body'] = df1['body'].str.replace(r'https?:\/\/\S*', '', regex=True)
    df1['body'] = df1['body'].str.replace(r'www.\S*', '', regex=True)
    df1['body_len'] = df1['body'].str.len()

    df1.to_csv(params().paths['boersen_articles']+params().filenames['boersen_merged'], encoding='utf-8', index=False)

    dtypes = {'id': 'bigint', 'headline': 'NVARCHAR(500)',
              'body': 'ntext', 'date': 'datetime2',
              'byline': 'NVARCHAR(500)', 'section_name': 'NVARCHAR(500)',
              'category': 'NVARCHAR(500)', 'byline_alt': 'NVARCHAR(500)',
              'body_len': 'int'}

    return df1, dtypes

df, _ = import_csv()

