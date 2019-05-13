__author__      = "Varada Kolhatkar"
import argparse
import sys, re
import numpy as np
import pandas as pd
import math

from spacy_features import CommentLevelFeatures

sys.path.append('../../')
from config import Config

sys.path.append('../modeling/')
from data_preprocessing_and_representation_for_deep_learning import *

sys.path.append(Config.PROJECT_HOME + 'source/feature_extraction/COMMENTIQ_code_subset/')
import commentIQ_features

class FeatureExtractor():
    '''
    A feature extractor for feature-based models.
    Extract comment features and write csvs with features
    '''

    def __init__(self, data_df, comment_column = 'pp_comment_text', label_column='constructive'):
        '''
        '''
        # Read all files
        print('data_df', data_df.shape)
        print('comment_column: ', comment_column)
        self.vocab = get_glove_dictionary()
        self.conjuctions_and_connectives = self.file2list(Config.RESOURCES_HOME + 'connectives.txt')
        self.stance_adverbials = self.file2list(Config.RESOURCES_HOME + 'stance_adverbials.txt')
        self.reasoning_verbs = self.file2list(Config.RESOURCES_HOME + 'reasoning_verbs.txt')
        self.root_clauses = self.file2list(Config.RESOURCES_HOME + 'root_clauses.txt')
        self.shell_nouns = self.file2list(Config.RESOURCES_HOME + 'shell_nouns.txt')
        self.modals = self.file2list(Config.RESOURCES_HOME + 'modals.txt')
        #self.data_csv = data_csv
        #self.df = pd.read_csv(data_csv, skiprows = [num for num in range(2000,50000)])
        #self.df = pd.read_csv(data_csv)
        self.df = data_df
        print(self.df.columns)
        print(self.df.shape)
        self.comment_col = comment_column
        self.features_data = []
        self.cols = []

    def get_features_df(self):
        '''
        :return:
        '''
        return self.df

    def file2list(self, file_name):
        '''
        :param file_name: String. Path of a filename
        :return: list

        Description: Given the file_name path, this function returns the values in the file as a list.

        '''
        L = open(file_name).readlines()
        L = [s.strip() for s in L]
        return L

    def has_conjunctions_and_connectives(self, comment):
        '''
        :param comment: String. The comment for feature extraction.
        :return: int
        '''
        if any(c in comment for c in self.conjuctions_and_connectives):
            #print('Comment with conjunctions and connectives: ', comment)
            return 1

        return 0
    
    def has_reasoning_verbs(self, comment):
        '''
        :param comment: String. The comment for feature extraction.
        :return: int
        '''
        if any(c in comment for c in self.reasoning_verbs):
            #print('Comment with reasoning verbs: ', comment)
            return 1
        
        return 0

    # Text quality features 
    def get_n_oov_words(self, comment):
        '''
        '''
        count = 0
        for word in comment.split():
            if word not in self.vocab:
                count+=1
        return count
    
    def get_n_caps(self, comment):
        '''
        '''
        words = comment.split()
        count = 0
        for word in words:
            if word.isupper():
                count+=1 
        return count
    
    def get_length(self, comment):
        '''
        :param comment: String. The comment for feature extraction.
        :return: int
        '''
        return len(comment.split())

    def get_average_word_length(self, comment):
        '''
        :param comment: String. The comment for feature extraction.
        :return: float
        '''
        return round(np.mean([len(word) for word in comment.split()]),3)

    def has_modals(self, comment):
        '''
        :param comment: String. The comment for feature extraction.
        :return: int
        '''
        if any(c in comment for c in self.modals):
            #print('Comment with modals: ', comment)
            return 1
        return 0

    def has_shell_nouns(self, comment):
        '''
        :param comment: String. The comment for feature extraction.
        :return: int
        '''
        if any(c in comment for c in self.shell_nouns):
            #print('Comment with shell nouns: ', comment)
            return 1

        return 0

    def has_stance_adverbials(self, comment):
        '''
        :param comment: String. The comment for feature extraction.
        :return: int
        '''
        if any(rc in comment for rc in self.stance_adverbials):
            #print('Comment with stance adverbials: ', comment)
            return 1
        return 0


    def has_root_clauses(self, comment):
        '''
        :param comment: String. The comment for feature extraction.
        :return: int
        '''
        if any(rc in comment for rc in self.root_clauses):
            #print('Comment with root clauses: ', comment)
            return 1
        return 0

    def has_characteristic(self, crowd_annotated_chars, characteristic):
        '''
        :param column:
        :param characteristic:
        :return:
        '''

        try:
            if type(crowd_annotated_chars) != str and math.isnan(crowd_annotated_chars):
                return np.NaN
        except:
            print('crowd_annotated_chars: ', crowd_annotated_chars)
            sys.exit(0)

        crowd_chars = [re.sub(':\d+', '', crowd_char) for crowd_char in crowd_annotated_chars.split()]
        if characteristic in crowd_chars:
            return 1
        return 0

    def get_comment_level_features(self, comment):
        '''
        :param comment: String. Comment to extract features.
        :return: int, int, float
        '''
        cf = CommentLevelFeatures(comment)

        # Named entity counts
        ner_count = len(cf.get_named_entities())

        # Number of sentences
        nsentscount = cf.get_sentence_counts()

        # Average nwords per sentence
        anwords = cf.average_nwords_per_sentence()

        #pos sequence for the comment
        pos_seq = cf.get_pos_seq()
        return ner_count, nsentscount, anwords, pos_seq

    def extract_crowd_annotated_features(self, char_cols = ['constructive_characteristics',
                                                 'non_constructive_characteristics',
                                                 'toxicity_characteristics'
                                                 ]):
        '''
        :param output_csv:
        :return:
        '''
        constructive_chars = ['specific_points', 'dialogue', 'evidence', 'personal_story', 'solution', 'no_con']
        non_constructive_chars = ['no_respect', 'provocative', 'sarcastic', 'non_relevant', 'unsubstantial', 'no_non_con']
        toxic_chars = ['personal_attack', 'teasing', 'abusive', 'embarrassment', 'inflammatory', 'no_toxic']

        print(self.df.columns)
        for char_col in char_cols:
            if char_col.startswith('constructive'):
                for char in constructive_chars:
                    self.df[char] = self.df[char_col].apply(self.has_characteristic, args = (char,))
            if char_col.startswith('non_constructive'):
                for char in non_constructive_chars:
                    self.df[char] = self.df[char_col].apply(self.has_characteristic, args = (char,))
            if char_col.startswith('toxic'):
                for char in toxic_chars:
                    self.df[char] = self.df[char_col].apply(self.has_characteristic, args = (char,))
        print(self.df.columns)

        #self.df['constructive'] = self.df['constructive'].apply(lambda x: 1 if x > Config.CONSTRUCTIVENESS_THRESHOLD else 0)
        #self.df.rename(columns={'comment_text':'pp_comment_text'}, inplace=True)
        #self.df.to_csv(output_csv, columns = cols, index = False)
        #print('Features CSV written: ', output_csv)


    def extract_features(self):
        '''
        :param output_csv: String. The CSV path to write feature vectors
        :return: None

        Description: Given the output CSV file path, output_csv, this function extracts features and writes
        them in output_csv.
        '''
        
        self.df['has_conjunctions_and_connectives'] = self.df[self.comment_col].apply\
            (self.has_conjunctions_and_connectives)

        self.df['has_stance_adverbials'] = self.df[self.comment_col].apply(
            self.has_stance_adverbials)

        self.df['has_reasoning_verbs'] = self.df[self.comment_col].apply(
            self.has_reasoning_verbs)

        self.df['has_modals'] = self.df[self.comment_col].apply(
            self.has_modals)

        self.df['has_shell_nouns'] = self.df[self.comment_col].apply(
            self.has_shell_nouns)

        self.df['length'] = self.df[self.comment_col].apply(
            self.get_length)

        self.df['average_word_length'] = self.df[self.comment_col].apply(
            self.get_average_word_length)

        self.df['ncaps'] = self.df[self.comment_col].apply(self.get_n_caps)
        self.df['noov'] = self.df[self.comment_col].apply(self.get_n_oov_words)        
        
        self.df['readability_score'] = self.df[self.comment_col].apply(
            commentIQ_features.calcReadability)

        self.df['personal_exp_score'] = self.df[self.comment_col].apply(
            commentIQ_features.calcPersonalXPScores)


        self.df['named_entity_count'], self.df['nSents'], self.df['avg_words_per_sent'], self.df['pos'] = \
            zip(*self.df[self.comment_col].apply(self.get_comment_level_features))

    def rename_column_names(self, rename_dict):
        self.df = self.df.rename(columns = rename_dict)
        #self.df['constructive'] = '?'
        #self.df['source'] = 'SOCC'
        
        
    def write_features_csv(self, output_csv,
                           cols = ['source',
                                   'comment_counter',                                    
                                   'pp_comment_text', 
                                   'constructive', 
                                   'njudgements_constructiveness_expt',
                                   'njudgements_toxicity_expt',
                                   'has_conjunctions_and_connectives',
                                   'has_stance_adverbials', 
                                   'has_reasoning_verbs', 
                                   'has_modals',
                                   'has_shell_nouns',
                                   'length', 
                                   'average_word_length', 
                                   'readability_score', 
                                   'personal_exp_score',
                                   'named_entity_count',
                                   'nSents',
                                   'avg_words_per_sent',
                                   'specific_points', 
                                   'dialogue', 
                                   'no_con',
                                   'evidence',
                                   'personal_story', 
                                   'solution',
                                   'no_respect', 
                                   'no_non_con',
                                   'provocative', 
                                   'sarcastic', 
                                   'non_relevant',
                                   'unsubstantial', 
                                   'personal_attack', 
                                   'teasing', 
                                   'no_toxic', 
                                   'abusive',
                                   'embarrassment', 
                                   'inflammatory'
                                   ]):
        '''
        :param cols:
        :return:
        '''
        print('colums:', cols)
        rename_dict = {'comment_text':'pp_comment_text',
            'SEVERE_TOXICITY_probability':'SEVERE_TOXICITY:probability', 
            'SEXUALLY_EXPLICIT_probability':'SEXUALLY_EXPLICIT:probability',
            'TOXICITY_probability':'TOXICITY:probability', 
            'TOXICITY_IDENTITY_HATE_probability':'TOXICITY_IDENTITY_HATE:probability',
            'TOXICITY_INSULT_probability':'TOXICITY_INSULT:probability', 
            'TOXICITY_OBSCENE_probability':'TOXICITY_OBSCENE:probability',
            'TOXICITY_THREAT_probability':'TOXICITY_THREAT:probability', 
            'ATTACK_ON_AUTHOR_probability':'ATTACK_ON_AUTHOR:probability',
            'ATTACK_ON_COMMENTER_probability':'ATTACK_ON_COMMENTER:probability', 
            'ATTACK_ON_PUBLISHER_probability':'ATTACK_ON_PUBLISHER:probability',
            'INCOHERENT_probability':'INCOHERENT:probability',
            'INFLAMMATORY_probability':'INFLAMMATORY:probability',
            'LIKELY_TO_REJECT_probability':'LIKELY_TO_REJECT:probability',
            'OBSCENE_probability':'OBSCENE:probability',
            'OFF_TOPIC_probability':'OFF_TOPIC:probability', 
            'SPAM_probability':'SPAM:probability',
            'UNSUBSTANTIAL_probability':'UNSUBSTANTIAL:probability'
          }
        clms = self.df.columns
        
        for c in cols:
            if c not in clms:
                self.df[c] = np.NaN
        print(self.df.columns)
        
        self.df = self.df.rename(columns = rename_dict, inplace = True)
        self.df.to_csv(output_csv, columns= cols, index = False)
        print(self.df.columns)
        print('Features CSV written: ', output_csv)

def get_arguments():
    parser = argparse.ArgumentParser(description='Constructiveness Feature Extractor')

    parser.add_argument('--train_dataset_path', '-tr', type=str, dest='train_data_path', action='store',
                        #default= Config.TRAIN_PATH + 'SOCC_NYT_picks_constructive_YNACC_non_constructive.csv',
                        default= Config.SOCC_ANNOTATED_WITH_PERSPECTIVE_SCORES, 
                        help="The path for the training data CSV.")

    parser.add_argument('--test_dataset_path', '-te', type=str, dest='test_data_path', action='store',
                        default= Config.TEST_PATH + 'SOCC_constructiveness.csv',
                        help="The path for the training data CSV.")

    parser.add_argument('--train_features_csv', '-trf', type=str, dest='train_features_csv', action='store',
                        #default= Config.TRAIN_PATH + 'SOCC_nyt_ync_features.csv',
                        default= Config.TRAIN_PATH + 'system_crowd_perspective_features.csv',
                        help="The file containing comments and extracted features for training data")

    parser.add_argument('--test_features_csv', '-tef', type=str, dest='test_features_csv', action='store',
                        default = Config.TEST_PATH + 'features.csv',
                        help="The file containing comments and extracted features for test data")

    args = parser.parse_args()
    return args

def extract_feats_for_old_annotated_SOCC():
    old_SOCC_constructiveness_df = pd.read_csv(Config.SOCC_ANNOTATED_CONSTRUCTIVENESS_1000)
    old_SOCC_constructiveness_df_subset = old_SOCC_constructiveness_df[['article_id', 'comment_counter', 'is_constructive', 'toxicity_level']]
    
    SOCC_with_perspective_scores_df = pd.read_csv(Config.SOCC_COMMENTS_WITH_PERSPECTIVE_SCORES)
    SOCC_with_perspective_scores_df = SOCC_with_perspective_scores_df[['comment_counter', 'comment_text', 'SEVERE_TOXICITY_probability', 'SEXUALLY_EXPLICIT_probability',
       'TOXICITY_probability', 'TOXICITY_IDENTITY_HATE_probability',
       'TOXICITY_INSULT_probability', 'TOXICITY_OBSCENE_probability',
       'TOXICITY_THREAT_probability', 'ATTACK_ON_AUTHOR_probability',
       'ATTACK_ON_COMMENTER_probability', 'ATTACK_ON_PUBLISHER_probability',
       'INCOHERENT_probability', 'INFLAMMATORY_probability',
       'LIKELY_TO_REJECT_probability', 'OBSCENE_probability',
       'OFF_TOPIC_probability', 'SPAM_probability',
       'UNSUBSTANTIAL_probability']]
    
    SOCC_with_perspective_scores_df_subset = SOCC_with_perspective_scores_df[SOCC_with_perspective_scores_df['comment_counter'].isin(old_SOCC_constructiveness_df_subset['comment_counter'].tolist())]
    
    df_merged = old_SOCC_constructiveness_df_subset.merge(SOCC_with_perspective_scores_df_subset, 
                                                       on=['comment_counter'], 
                                                       how='inner')
    
    df_merged['constructive'] = df_merged['is_constructive'].apply(lambda x: 1.0 if x == 'yes' else 0.)
    
    fe_train = FeatureExtractor(df_merged, comment_column = 'comment_text')
    fe_train.extract_features()    

    
    rename_dict = {'comment_text':'pp_comment_text',
                'SEVERE_TOXICITY_probability':'SEVERE_TOXICITY:probability', 
                'SEXUALLY_EXPLICIT_probability':'SEXUALLY_EXPLICIT:probability',
                'TOXICITY_probability':'TOXICITY:probability', 
                'TOXICITY_IDENTITY_HATE_probability':'TOXICITY_IDENTITY_HATE:probability',
                'TOXICITY_INSULT_probability':'TOXICITY_INSULT:probability', 
                'TOXICITY_OBSCENE_probability':'TOXICITY_OBSCENE:probability',
                'TOXICITY_THREAT_probability':'TOXICITY_THREAT:probability', 
                'ATTACK_ON_AUTHOR_probability':'ATTACK_ON_AUTHOR:probability',
                'ATTACK_ON_COMMENTER_probability':'ATTACK_ON_COMMENTER:probability', 
                'ATTACK_ON_PUBLISHER_probability':'ATTACK_ON_PUBLISHER:probability',
                'INCOHERENT_probability':'INCOHERENT:probability',
                'INFLAMMATORY_probability':'INFLAMMATORY:probability',
                'LIKELY_TO_REJECT_probability':'LIKELY_TO_REJECT:probability',
                'OBSCENE_probability':'OBSCENE:probability',
                'OFF_TOPIC_probability':'OFF_TOPIC:probability', 
                'SPAM_probability':'SPAM:probability',
                'UNSUBSTANTIAL_probability':'UNSUBSTANTIAL:probability'
              }
    
    cols = ['article_id', 'comment_counter', 'pp_comment_text', 'constructive', 
        'has_conjunctions_and_connectives',
        'has_stance_adverbials', 'has_reasoning_verbs', 'has_modals', 'has_shell_nouns',
        'length', 'average_word_length', 'readability_score', 'personal_exp_score',
        'named_entity_count', 'nSents', 'avg_words_per_sent',             
        'SEVERE_TOXICITY:probability', 'SEXUALLY_EXPLICIT:probability',
        'TOXICITY:probability', 'TOXICITY_IDENTITY_HATE:probability',
        'TOXICITY_INSULT:probability', 'TOXICITY_OBSCENE:probability',
        'TOXICITY_THREAT:probability', 'ATTACK_ON_AUTHOR:probability',
        'ATTACK_ON_COMMENTER:probability', 'ATTACK_ON_PUBLISHER:probability',
        'INCOHERENT:probability', 'INFLAMMATORY:probability',
        'LIKELY_TO_REJECT:probability', 'OBSCENE:probability',
        'OFF_TOPIC:probability', 'SPAM:probability',
        'UNSUBSTANTIAL:probability'
       ]
    fe_train.rename_column_names(rename_dict)
    fe_train.write_features_csv(Config.TRAIN_PATH + 'SOCC_old_constructiveness_annotations_feats.csv', cols = cols)
    #df_merged.rename(columns = rename_dict, inplace=True)
    #df_merged.to_csv(Config.TRAIN_PATH + 'SOCC_old_constructiveness_annotations_feats.csv', columns = cols, index = False) 


if __name__ == "__main__":
    #extract_feats_for_old_annotated_SOCC()
    #df = pd.read_csv(Config.SOCC_ANNOTATED_CONSTRUCTIVENESS_12000)
    #df_clean = df[(df['constructive'] < 0.4) | (df['constructive'] > 0.6)]
    #print(df_clean.shape)
    #fe_train = FeatureExtractor(df_clean, comment_column = 'comment_text')        
    #fe_train.extract_features()  
    #fe_train.extract_crowd_annotated_features()
    #feats_df = fe_train.get_features_df()
    #feats_df = feats_df.rename(columns={'comment_text':'pp_comment_text'})
    #feats_df.to_csv(Config.TRAIN_PATH + 'CTC_crowd_annotated_feats.csv', index = False) 
    crowd_feats_df = pd.read_csv(Config.TRAIN_PATH + 'CTC_crowd_annotated_feats.csv')
    cf_df = crowd_feats_df[['comment_counter', 'pp_comment_text', 'specific_points', 
                        'dialogue', 'evidence', 'personal_story', 'solution',
                        'no_con', 'no_respect', 'provocative', 'sarcastic', 'non_relevant',
                        'unsubstantial', 'no_non_con', 'personal_attack', 'teasing', 'abusive',
                        'embarrassment', 'inflammatory', 'no_toxic']]
    SOCC_other_feats_df = pd.read_csv(Config.ALL_SOCC_FEATURES_FILE_PATH)
    SOCC_df = SOCC_other_feats_df[SOCC_other_feats_df['source'] == 'SOCC']
    print(SOCC_df.columns)
    result_df = cf_df.merge(SOCC_df, on=['comment_counter', 'pp_comment_text'], how='inner')
    result_df.to_csv(Config.TRAIN_PATH + 'CTC_with_all_feats.csv')
    sys.exit(0)
    
    # Extract features for SOCC
    df = pd.read_csv(Config.ALL_SOCC_FEATURES_FILE_PATH)
    print(df.columns)
    df_clean = df.drop(df[(df['constructive'] >= 0.4) & (df['constructive'] <= 0.6)].index)    
    fe_train = FeatureExtractor(df_clean, comment_column = 'pp_comment_text')
    fe_train.extract_features()    
    feats_df = fe_train.get_features_df()
    cols = ['comment_counter', 'pp_comment_text', 'constructive', 'source', 
        'has_conjunctions_and_connectives',
        'has_stance_adverbials', 'has_reasoning_verbs', 'has_modals', 'has_shell_nouns',
        'length', 'average_word_length', 'readability_score', 'personal_exp_score',
        'named_entity_count', 'nSents', 'avg_words_per_sent',             
        'pos', 'ncaps', 'noov',         
        'SEVERE_TOXICITY:probability', 'SEXUALLY_EXPLICIT:probability',
        'TOXICITY:probability', 'TOXICITY_IDENTITY_HATE:probability',
        'TOXICITY_INSULT:probability', 'TOXICITY_OBSCENE:probability',
        'TOXICITY_THREAT:probability', 'ATTACK_ON_AUTHOR:probability',
        'ATTACK_ON_COMMENTER:probability', 'ATTACK_ON_PUBLISHER:probability',
        'INCOHERENT:probability', 'INFLAMMATORY:probability',
        'LIKELY_TO_REJECT:probability', 'OBSCENE:probability',
        'OFF_TOPIC:probability', 'SPAM:probability',
        'UNSUBSTANTIAL:probability'
       ]    
    feats_df.to_csv(Config.TRAIN_PATH + 'SOCC_all_feats.csv', columns = cols, index = False)    
    
    sys.exit(0)

    
    # Extract features for sentences training data
    df = pd.read_csv(Config.TRAIN_PATH + 'constructive_sentences_training_data.csv')
    fe_train = FeatureExtractor(df, comment_column = 'sentence')
    fe_train.extract_features()
    features_df = fe_train.get_features_df()
    features_df.to_csv(Config.TRAIN_PATH + 'SOCC_sents_features.csv', index = False)
    #fe_train.write_features_csv(Config.TRAIN_PATH + 'SOCC_sents_features.csv')  
    sys.exit(0)
    
    # For training only get instances where the constructiveness label is reliable 
    df = pd.read_csv(Config.SOCC_COMMENTS_WITH_PERSPECTIVE_SCORES)
    df_clean = df.drop(df[(df['constructive'] >= 0.4) & (df['constructive'] <= 0.6)].index)
    print(df_clean.columns)
    fe_train = FeatureExtractor(df_clean, comment_column = 'comment_text')
    fe_train.extract_features()    
    cols = ['comment_counter', 'pp_comment_text', 'constructive', 'source', 
        'has_conjunctions_and_connectives',
        'has_stance_adverbials', 'has_reasoning_verbs', 'has_modals', 'has_shell_nouns',
        'length', 'average_word_length', 'readability_score', 'personal_exp_score',
        'named_entity_count', 'nSents', 'avg_words_per_sent',             
        'SEVERE_TOXICITY:probability', 'SEXUALLY_EXPLICIT:probability',
        'TOXICITY:probability', 'TOXICITY_IDENTITY_HATE:probability',
        'TOXICITY_INSULT:probability', 'TOXICITY_OBSCENE:probability',
        'TOXICITY_THREAT:probability', 'ATTACK_ON_AUTHOR:probability',
        'ATTACK_ON_COMMENTER:probability', 'ATTACK_ON_PUBLISHER:probability',
        'INCOHERENT:probability', 'INFLAMMATORY:probability',
        'LIKELY_TO_REJECT:probability', 'OBSCENE:probability',
        'OFF_TOPIC:probability', 'SPAM:probability',
        'UNSUBSTANTIAL:probability'
       ]
    
    sys.exit(0)
    fe_train.write_features_csv(Config.TRAIN_PATH + 'SOCC_features.csv', cols = SOCC_cols)  
    args = get_arguments()    
    print(args)
    cols = ['pp_comment_text', 'constructive', 'source', 'crowd_toxicity_level', 
            'specific_points', 'dialogue', 'no_con',
            'evidence', 'personal_story', 'solution', 'no_respect', 'no_non_con',
            'provocative', 'sarcastic', 'non_relevant',
            'unsubstantial', 'personal_attack', 'teasing', 'no_toxic', 'abusive',
            'embarrassment', 'inflammatory', 'has_conjunctions_and_connectives',
            'has_stance_adverbials', 'has_reasoning_verbs', 'has_modals', 'has_shell_nouns',
            'length', 'average_word_length', 'readability_score', 'personal_exp_score',
            'named_entity_count', 'nSents', 'avg_words_per_sent',             
            'SEVERE_TOXICITY:probability', 'SEXUALLY_EXPLICIT:probability',
            'TOXICITY:probability', 'TOXICITY_IDENTITY_HATE:probability',
            'TOXICITY_INSULT:probability', 'TOXICITY_OBSCENE:probability',
            'TOXICITY_THREAT:probability', 'ATTACK_ON_AUTHOR:probability',
            'ATTACK_ON_COMMENTER:probability', 'ATTACK_ON_PUBLISHER:probability',
            'INCOHERENT:probability', 'INFLAMMATORY:probability',
            'LIKELY_TO_REJECT:probability', 'OBSCENE:probability',
            'OFF_TOPIC:probability', 'SPAM:probability',
            'UNSUBSTANTIAL:probability'
           ]
    
    #fe_train.write_features_csv(args.train_features_csv, cols)    
    fe_train = FeatureExtractor(pd.read_csv(Config.SOCC_COMMENTS_WITH_PERSPECTIVE_SCORES), comment_column = 'comment_text')
    fe_train.extract_features()    
    #fe_train.extract_crowd_annotated_features()
    
    rename_dict = { 'comment_text':'pp_comment_text',
                    'SEVERE_TOXICITY_probability':'SEVERE_TOXICITY:probability', 
                    'SEXUALLY_EXPLICIT_probability':'SEXUALLY_EXPLICIT:probability',
                    'TOXICITY_probability':'TOXICITY:probability', 
                    'TOXICITY_IDENTITY_HATE_probability':'TOXICITY_IDENTITY_HATE:probability',
                    'TOXICITY_INSULT_probability':'TOXICITY_INSULT:probability', 
                    'TOXICITY_OBSCENE_probability':'TOXICITY_OBSCENE:probability',
                    'TOXICITY_THREAT_probability':'TOXICITY_THREAT:probability', 
                    'ATTACK_ON_AUTHOR_probability':'ATTACK_ON_AUTHOR:probability',
                    'ATTACK_ON_COMMENTER_probability':'ATTACK_ON_COMMENTER:probability', 
                    'ATTACK_ON_PUBLISHER_probability':'ATTACK_ON_PUBLISHER:probability',
                    'INCOHERENT_probability':'INCOHERENT:probability',
                    'INFLAMMATORY_probability':'INFLAMMATORY:probability',
                    'LIKELY_TO_REJECT_probability':'LIKELY_TO_REJECT:probability',
                    'OBSCENE_probability':'OBSCENE:probability',
                    'OFF_TOPIC_probability':'OFF_TOPIC:probability', 
                    'SPAM_probability':'SPAM:probability',
                    'UNSUBSTANTIAL_probability':'UNSUBSTANTIAL:probability'
                  }
    fe_train.rename_column_names(rename_dict)
    
    SOCC_cols = ['comment_counter', 'pp_comment_text', 'constructive', 'source', 
        'has_conjunctions_and_connectives',
        'has_stance_adverbials', 'has_reasoning_verbs', 'has_modals', 'has_shell_nouns',
        'length', 'average_word_length', 'readability_score', 'personal_exp_score',
        'named_entity_count', 'nSents', 'avg_words_per_sent',             
        'SEVERE_TOXICITY:probability', 'SEXUALLY_EXPLICIT:probability',
        'TOXICITY:probability', 'TOXICITY_IDENTITY_HATE:probability',
        'TOXICITY_INSULT:probability', 'TOXICITY_OBSCENE:probability',
        'TOXICITY_THREAT:probability', 'ATTACK_ON_AUTHOR:probability',
        'ATTACK_ON_COMMENTER:probability', 'ATTACK_ON_PUBLISHER:probability',
        'INCOHERENT:probability', 'INFLAMMATORY:probability',
        'LIKELY_TO_REJECT:probability', 'OBSCENE:probability',
        'OFF_TOPIC:probability', 'SPAM:probability',
        'UNSUBSTANTIAL:probability'
       ]
        
    fe_train.write_features_csv(Config.TRAIN_PATH + 'SOCC_features.csv', cols = SOCC_cols)
    #nyt_ync_df = fe_train.get_features_df()
    #print('columns: ', nyt_ync_df.columns)
    
     #fe_test = FeatureExtractor(args.test_data_path)
    #fe_test.extract_features(args.test_features_csv)



