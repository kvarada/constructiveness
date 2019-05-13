# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

def preprocess(text):
    text = text.strip()
    text = re.sub('\s+', ' ', text)
    text = re.sub(r'[\"\']', '', text)    
    text = " ".join(word_tokenize(text))
    return text

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    df = pd.read_csv(input_filepath)
    df['constructive_binary'] = df['constructive'].apply(lambda x: 1.0 if x > 0.5 else 0.0)
    df['pp_comment_text'] = df['comment_text'].apply(preprocess)    
    df.to_csv(output_filepath)
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    
    main()
    #print(input_filepath)