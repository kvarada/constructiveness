#!/usr/local/bin/python3
__author__      = "Varada Kolhatkar"
import pandas as pd
import numpy as np
from normalize_comments import *
COMMENT_WORDS_THRESHOLD = 4
CONSTRUCTIVENESS_SCORE_THRESHOLD = 0.6

class ConstructivenessDataCollector:
    '''
    A class to collect training and test data for constructiveness
    from different resources
    '''

    def __init__(self):
        '''
        '''
        # initialize a dataframe for the training data
        self.training_df = pd.DataFrame(columns=['comment_text', 'constructive', 'source'])
        self.test_df = pd.DataFrame(columns=['comment_counter', 'comment_text', 'constructive'])
        self.training_df_normalized = None
        self.test_df_normalized = None

    def get_positive_examples(self):
        '''
        :return:
        '''
        positive_df = self.training_df[self.training_df['constructive'] == 1]
        return positive_df

    def get_negative_examples(self):
        '''
        :return:
        '''
        negative_df = self.training_df[self.training_df['constructive'] == 0]
        return negative_df

    def normalize_comment_text(self,  column = 'comment_text', mode = 'train'):
        #start = timer()
        #print('Start time: ', start)
        if mode.startswith('train'):
            df = self.training_df
        else:
            df = self.test_df

        df_processed = parallelize(df, run_normalize)
        #end = timer()
        #print('Total time taken: ', end - start)

        if mode.startswith('train'):
            self.training_df_normalized = df_processed
        else:
            self.test_df_normalized = df_processed

    def collect_training_data_from_CSV(self, data_csv, frac = 1.0, source = 'SOCC',
                                       cols_dict={'constructive': 'constructive',
                                                  'comment_text': 'comment_text',
                                                  'comment_word_count': 'commentWordCount'}):
        '''
        :param data_csv:
        :param frac:
        :param cols_dict:
        :return:
        '''
        df = pd.read_csv(data_csv, skipinitialspace=True)
        df = df.sample(frac = frac)
        if not cols_dict['comment_word_count'] in df:
            df[cols_dict['comment_word_count']] = df[cols_dict['comment_text']].apply(lambda x: len(x.split()))

        df.rename(columns={cols_dict['comment_text']: 'comment_text',
                           cols_dict['comment_word_count']:'comment_word_count',
                           cols_dict['constructive']: 'constructive'}, inplace = True)
        df['source'] = source
        # Select comments selected by NYT moderators as NYT pick and where the length
        # of the comment is  > COMMENT_WORDS_THRESHOLD
        df['constructive'] = df['constructive'].apply(lambda x: 1 if x > CONSTRUCTIVENESS_SCORE_THRESHOLD else 0)
        self.training_df = pd.concat([self.training_df, df[['comment_text', 'constructive', 'source', 
                                                            'crowd_toxicity_level',
                                                            'constructive_characteristics',
                                                            'non_constructive_characteristics',
                                                            'toxicity_characteristics']]])
        self.normalize_comment_text(mode='train')
        #self.write_csv(output_csv)

    def collect_positive_examples(self, positive_examples_csv, frac = 1.0, source = 'NYTPicks',
                                  cols_dict = {'constructive':'editorsSelection',
                                               'comment_text':'commentBody',
                                               'comment_word_count': 'commentWordCount'}):
        '''
        :param positive_examples_csv:
        :param frac:
        :param cols_dict:
        :return:
        '''
        df = pd.read_csv(positive_examples_csv, skipinitialspace=True)
        df = df.sample(frac = frac)

        if not cols_dict['comment_word_count'] in df:
            df[cols_dict['comment_word_count']] = df[cols_dict['comment_text']].apply(lambda x: len(x.split()))

        df.rename(columns={cols_dict['comment_text']: 'comment_text',
                           cols_dict['comment_word_count']:'comment_word_count',
                           cols_dict['constructive']: 'constructive'}, inplace = True)
        df['source'] = source
        df['crowd_toxicity_level'] = np.NaN
        df['constructive_characteristics'] = np.NaN
        df['non_constructive_characteristics'] = np.NaN
        df['toxicity_characteristics'] = np.NaN
        # Select comments selected by NYT moderators as NYT pick and where the length
        # of the comment is  > COMMENT_WORDS_THRESHOLD
        positive_df = df[
            (df['constructive'] == 1) & (df['comment_word_count'] > COMMENT_WORDS_THRESHOLD)]
        self.training_df = pd.concat([self.training_df, positive_df[['comment_text', 'constructive', 'source',
                                                                     'crowd_toxicity_level', 
                                                                     'constructive_characteristics',
                                                                     'non_constructive_characteristics',
                                                                     'toxicity_characteristics'
                                                                     ]]])
        self.normalize_comment_text(mode='train')
        return positive_df

    def collect_negative_examples(self, negative_examples_csv, frac = 1.0, source = 'YNACC',
                                  cols_dict = {'comment_text': 'text',
                                               'constructive': 'constructiveclass'}):
        '''
        :param negative_examples_csv:
        :param frac:
        :param cols_dict:
        :return:
        '''

        if negative_examples_csv.endswith('tsv'):
            df = pd.read_csv(negative_examples_csv, sep='\t')
        else:
            df = pd.read_csv(negative_examples_csv)

        df = df.sample(frac = frac)
        df.rename(columns={cols_dict['comment_text']: 'comment_text'}, inplace=True)
        df['comment_word_count'] = df['comment_text'].apply(lambda x: len(x.split()))
        df['source'] = source
        df['crowd_toxicity_level'] = np.NaN        
        df['constructive_characteristics'] = np.NaN
        df['non_constructive_characteristics'] = np.NaN
        df['toxicity_characteristics'] = np.NaN

        df_subset = df[(df['comment_word_count'] > COMMENT_WORDS_THRESHOLD) & (
            df[cols_dict['constructive']].str.startswith('Not'))]
        negative_df = df_subset.copy()

        negative_df['constructive'] = negative_df[cols_dict['constructive']].apply(lambda x: 0 if x.startswith('Not') else 1)
        self.training_df = pd.concat([self.training_df, negative_df[['comment_text', 'constructive', 'source',
                                                                     'crowd_toxicity_level', 
                                                                     'constructive_characteristics',
                                                                     'non_constructive_characteristics',
                                                                     'toxicity_characteristics'
                                                                     ]]])
        return negative_df

    def gather_test_data(self, test_data_file, frac = 1.0,
                          cols_dict={'comment_text': 'comment_text',
                                     'constructive': 'is_constructive'}):
        '''
        :param test_data_file:
        :param frac:
        :param cols_dict:
        :return:
        '''
        df = pd.read_csv(test_data_file)
        df = df.sample(frac = frac)
        df.rename(columns={cols_dict['comment_text']:'comment_text'}, inplace = True)
        df['constructive'] = df[cols_dict['constructive']].apply(lambda x: 1 if x.startswith('yes') else 0)
        df = df.sort_values(by=['constructive'], ascending = False)
        self.test_df = pd.concat([self.test_df, df[['comment_counter', 'comment_text', 'constructive']]])

    def collect_train_data(self,
            positive_examples_csvs,
            negative_examples_csvs):
        '''
        :param positive_examples_csvs:
        :param negative_examples_csvs:
        :return:
        '''

        ## Collect Training data

        # Collect non-constructive examples to the training data
        for (source_csv, fraction) in negative_examples_csvs:
            df = self.collect_negative_examples(source_csv, frac=fraction)
            print('The number of negative examples collected from source: \n', source_csv, '\nFraction: ', fraction,
                  '\n is: ', df.shape[0])
            print('---------------------------')

        for (source_csv, fraction) in positive_examples_csvs:
            df = self.collect_positive_examples(source_csv, frac = fraction)
            print('The number of positive examples collected from source: \n', source_csv, '\nFraction: ', fraction,
                  '\n is: ', df.shape[0])
            print('---------------------------')

        # Normalize comment text
        self.normalize_comment_text()

        negative_df = self.get_negative_examples()
        positive_df = self.get_positive_examples()
        print('\n\n')
        print('Total number of positive examples: ', positive_df.shape[0])
        print('Total number of negative examples: ', negative_df.shape[0])

    def collect_test_data(self, test_csvs, output_csv):
        '''
        :param test_csvs:
        :param output_csv:
        :return:
        '''
        # Collect test data
        for (test_csv_file, fraction) in test_csvs:
            self.gather_test_data(test_csv_file, frac=fraction)

        # Normalize the comments
        self.normalize_comment_text(mode='test')

        # Write test data CSV
        self.write_csv(output_csv, mode='test',
                      cols=['comment_counter', 'comment_text', 'pp_comment_text', 'constructive'])


    def write_csv(self, output_csv, mode ='train', cols = ['comment_text', 'pp_comment_text',
                                                           'constructive', 'source',
                                                           'crowd_toxicity_level', 
                                                           'constructive_characteristics',
                                                           'non_constructive_characteristics',
                                                           'toxicity_characteristics'], index = False):
        '''
        :param output_csv:
        :param mode:
        :param cols:
        :param index:
        :return:
        '''
        print('\n')
        if mode.startswith('train'):
            print('The number of training examples: ', self.training_df_normalized.shape[0])
            self.training_df_normalized.to_csv(output_csv, columns=cols, index = index)
            print('Training data written as a CSV: ', output_csv)
        elif mode.startswith('test'):
            print('The number of test examples: ', self.test_df_normalized.shape[0])
            self.test_df_normalized.to_csv(output_csv, columns=cols, index = index)
            print('Test data written as a CSV: ', output_csv)
        else:
            print('Invalid mode!!')


