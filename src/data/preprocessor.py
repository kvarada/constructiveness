import pandas as pd
import numpy as np
import os 
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from wordsegment import load, segment
load()
import numpy as np
import multiprocessing as mp
from multiprocessing import cpu_count
import re, time

class Preprocessor:
    
    def __init__(self):
        gf = open(os.environ['GLOVE_DICTIONARY_PATH'])
        self.glove_words = [w.strip() for w in gf.readlines()]
        
    def preprocess(self, text):
        """
        """    
        # Remove urls
        text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)    
        text = re.sub(r'www..*\/\/.*', '', text, flags=re.MULTILINE)            
        tokens = nltk.word_tokenize(text)
        preprocessed = []
        
        for token in tokens: 
            if token not in self.glove_words: 
                segmented_tokens = segment(token)
                preprocessed.extend(segmented_tokens)
            else:
                preprocessed.append(token)
                
        return " ".join(preprocessed)

def run_normalize(df):
    pp = Preprocessor()    
    df['pp_comment_text'] = df['comment_text'].apply(pp.preprocess)
    return df

def parallelize(df, func):
    data_split = np.array_split(df, cores)
    pool = mp.Pool(cores)
    df_processed = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return df_processed

if __name__=="__main__":
    C3_feats_df = pd.read_csv(os.environ['C3_FEATS'])
    start = time.time()
    print('Start time: ', start)
    cores = cpu_count()
    # Preprocess the rows in parallel 
    df_processed = parallelize(C3_feats_df, run_normalize)
    end = time.time()
    print('Total time taken: ', end-start)    
    df_processed.to_csv(os.environ['C3_FEATS_PREPROCESSED'], index = False)
    print('The preprocessed features file written: ', os.environ['C3_FEATS_PREPROCESSED'])

    