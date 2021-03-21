# -*- coding: utf-8 -*-

import gzip
import json
import os
import shutil
import datetime
import pickle
import codecs
#import h5py
import pandas as pd

def main_directory():
    """
    Returns root path of project/package
    """
    return os.path.join(os.path.abspath(__file__).split('FUI')[0],'FUI')

def dump_pickle(folder_path, file_name, df, verbose=False):
    """
    Function that pickles df in folder_path\file_name (creates folder_path if doesn't exist)
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(os.path.join(folder_path, file_name), 'wb') as f_out:
        pickle.dump(df, f_out)
        if verbose:
            print("Wrote file '{}' with shape {} to disc".format(file_name, df.shape))

def dump_hdf(folder_path, file_name, df, verbose=False):
    file_path = os.path.join(folder_path, file_name)
    df.to_hdf(file_path, 'table', format='table', mode='w', append=False)
    if verbose:
        print("Wrote file '{}' with shape {} to disc".format(file_name, df.shape))


def read_hdf(file_path, verbose=False, obj='table'):
    try:
        df = pd.read_hdf(file_path, obj)
        if verbose:
            print("Loaded pickle with {} rows".format(len(df)))
        return df
    except FileNotFoundError:
        print("File not found!")
        return None

def read_h5py(file_path, obj='parsed_strings'):
    try:
        hf = h5py.File(file_path, 'r')
        df = pd.DataFrame(hf['parsed_strings'][:])
        df.rename(columns={0: 'body',
                           1: 'byline',
                           2: 'byline_alt',
                           3: 'category',
                           4: 'date',
                           5: 'headline',
                           6: 'id',
                           7: 'section_name',
                           8: 'body_len',
                           9: 'year',
                           10: 'word_count',
                           11: 'article_id'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df['article_id'] = df['article_id'].astype('int')
        df['word_count'] = df['word_count'].astype('float')
        df['body_len'] = df['body_len'].astype('float')
        df['year'] = df['year'].astype('int')
        df['id'] = df['id'].astype('float')
        if len(df.columns) > 12:
            df.rename(columns={12: 'u_count'}, inplace=True)
            df['u_count'] = df['u_count'].astype('int')
            df.rename(columns={13:'n_count'}, inplace=True)
            df['n_count'] = df['n_count'].astype('int')
        return df
    except FileNotFoundError:
        print("File not found!")
        return None

def dump_csv(folder_path, file_name, df, verbose=False):
    """
    Function that outputs df as csv in folder_path\file_name (creates folder_path if doesn't exist)
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    df.to_csv(os.path.join(folder_path, file_name+'.csv'), sep=';')
    if verbose:
        print("Wrote file '{}' with shape {} to disc".format(file_name, df.shape))


def flatten(mylist):
    newlist = [item for sublist in mylist for item in sublist]
    return newlist


def get_files_list(folder, suffix=''):
    """
    Function that returns list of files in folder ending with *.suffix
    """
    if suffix:
        ls_out = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and
                  f[-len(suffix):] == suffix]
    else:
        ls_out = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    return sorted(ls_out)


def unzip_files(folder, suffix='.gz'):
    """
    A short utility for unzipping all files (e.g. .gz) in a folder
    """
    files_list = get_files_list(folder, suffix)
    for f in files_list:
        with gzip.open(os.path.join(folder, f), 'rb') as f_in:
            with open(os.path.join(folder, f)[:-len(suffix)], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def params(ls_paths=['scripts', 'input_params.json'], reset=False):
    """
    Function for loading parameters for the entire workflow relative to \\FUI\\
    """

    if reset is True:
        _Singleton._instance = None

    if _Singleton._instance is None:
        input_file = os.path.abspath(os.path.join(__file__, '..', '..', '..', *ls_paths))
        try:
            with codecs.open(input_file, 'r', encoding='utf-8-sig') as f:
                input_json = json.load(f)
                _Singleton._instance = _Singleton(input_json)
        except FileNotFoundError as e:
            print(e)
    return _Singleton._instance


class _Singleton:
    _instance = None

    def __init__(self, input_json):
        input_path = os.path.abspath(os.path.join(__file__, '..', '..', '..'))
        self.filenames = input_json['filenames']
        self.options = input_json['options']
        self.dicts = input_json['dicts']
        self.paths = {key: os.path.join(input_path, path) for key, path in input_json['paths'].items()}
        for _, folder_path in self.paths.items():
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

