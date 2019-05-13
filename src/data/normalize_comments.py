__author__ = 'Varada Kolhatkar'
import argparse, sys, os, glob, ntpath, pprint, codecs, re, csv
import pandas as pd
import html.entities
from timeit import default_timer as timer
import numpy as np
import multiprocessing as mp
from multiprocessing import cpu_count
import re

cores = cpu_count()
partitions = cores


def word_segmentation(text):
    '''
    :param text: string. Comment text for word segmentation
    :return: string. Returns a word-segmented comment text
    '''

    # Most of the problems in the comments are with respect to missing spaces after punctuations.
    # Tried more sofistocated segmetation but it added more noise than corrections. The following
    # simplistic word segmentation is enough for our data, which separates words on punctuation.
    # For example, this will take care of following examples:
    # Rich,This => Rich, This

    missing_space_obj = re.compile(r'(?P<prev_word>(^|\s)\w{3,25}[.,?!;:]{1,3})(?P<next_word>(\w+))')

    def repl(m):
        return m.group('prev_word') + ' ' + m.group('next_word')

    text = missing_space_obj.sub(repl, text)
    return text

def unescape(text):
    '''
    :param text:
    :return:
     Description
    ##
    # Removes HTML or XML character references and entities from a text string.
    #
    # @param text The HTML (or XML) source text.
    # @return The plain text, as a Unicode string, if necessary.
    '''
    def fixup(m):
        text = m.group(0)
        if text[:2] == "&#":
            # character reference
            try:
                if text[:3] == "&#x":
                    return chr(int(text[3:-1], 16))
                else:
                    return chr(int(text[2:-1]))
            except ValueError:
                pass
        else:
            # named entity
            try:
                text = chr(html.entities.name2codepoint[text[1:-1]])
            except KeyError:
                pass
        return text # leave as is
    return re.sub("&#?\w+;", fixup, text)

def clean_text(text):
    '''
    :param text: string. Text to be cleaned
    :return: sting. cleaned text
    '''
    # Some strings have bite string notation converted as an str. Getting rid of these notations.
    text = text.lstrip("b'")
    text = text.rstrip("'")
    if type(text) != str:
        text = text.decode("utf-8")
    text = text.replace(r'<br/>', " ")
    text = text.replace(r'<br\>', " ")
    text = text.replace(r'|', " ")
    text = " ".join(text.split())
    text = text.replace(r"`",r"'")

    # Some comments contain text from previous comments for which it has posted a reply. Get rid of this text from the
    # comment
    text = re.sub(r'\(In reply to:.*?--((\s\S+){1,10})?\)', '', text)
    return text

def normalize(text):
    '''
    :param text: string. Text to be normalized
    :return: string. Normalized text
    '''
    try:
        cleaned = clean_text(text)
        text_ws = word_segmentation(cleaned)
        text_preprocessed = unescape(text_ws)
    except:
        print('------------')
        print('Problem text: ', text)
        print('------------')
        text_preprocessed = text
    return text_preprocessed

def run_normalize(df):
    '''
    :param df:
    :return:
    '''
    df['pp_comment_text'] = df['comment_text'].apply(normalize)
    return df

def parallelize(data, func):
    data_split = np.array_split(data, partitions)
    pool = mp.Pool(cores)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data

def get_arguments():
    parser = argparse.ArgumentParser(description='SFU Sentiment Calculator')
    parser.add_argument('--train_dataset_path', '-tr', type=str, dest='train_data_path', action='store',
                        default = '/Users/vkolhatk/Data/Constructiveness/data/train/NYT_picks_constructive_ynacc_non_constructive.csv',
                        help="The training data csv")

    parser.add_argument('--test_dataset_path', '-te', type=str, dest='test_data_path', action='store',
                        default='/Users/vkolhatk/Data/Constructiveness/data/test/SFU_constructiveness_toxicity_corpus.csv',
                        help="The test dataset path for constructive and non-constructive comments")

    parser.add_argument('--features_csv', '-f', type=str, dest='features_csv', action='store',
                        help="The file containing comments and extracted features")


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_arguments()
    print(args)
    start = timer()
    print('Start time: ', start)
    df = pd.read_csv(args.train_data_path)
    df_processed = parallelize(df, run_normalize)
    df_processed.to_csv(args.train_data_path.rstrip('.csv') + '_preprocessed.csv', index=False)
    print('Output csv written: ', args.train_data_path.rstrip('.csv') + '_preprocessed.csv')
    end = timer()
    print('Total time taken: ', end-start)