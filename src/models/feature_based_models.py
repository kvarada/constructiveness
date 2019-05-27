import sys, os, re, argparse

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import SGDClassifier
import pickle
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

import matplotlib.pyplot as plt

# classifiers / models
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# other
from sklearn.preprocessing import normalize
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
import nltk
import time

from sklearn.metrics import f1_score, classification_report
from itertools import chain, combinations

class FBConstructivenessClassifier():
    '''
    '''
    def __init__(self,
                 X_train, y_train, X_valid, y_valid, 
                 comments_col = 'pp_comment_text',
                 label_col = 'constructive_binary'):
        """
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.comments_col = comments_col
        self.label_col = label_col
        
    def show_scores(self, model):
        """
        """
        print("Training accuracy:   %.2f" % (model.score(self.X_train, self.y_train)))
        print("Validation accuracy: %.2f" % (model.score(self.X_valid, self.y_valid)))
        predictions = list(model.predict(self.X_train))
        true_labels = self.y_train.tolist()
        print('TRAIN CLASSIFICATION REPORT\n\n', classification_report(true_labels, predictions))

        predictions = list(model.predict(self.X_valid))
        true_labels = self.y_valid.tolist()
        print('VALIDATION CLASSIFICATION REPORT\n\n', classification_report(true_labels, predictions))

    def sweep_hyper(self, model, hyper, val_range):
        """
        """
        train_errors = []
        valid_errors = []
        for val in val_range:
            m = model(**{hyper:val})
            m.fit(X_C3_train, y_C3_train)
            train_errors.append(1-m.score(X_C3_train, y_C3_train))
            valid_errors.append(1-m.score(X_C3_valid, y_C3_valid))
        plt.semilogx(C_range, train_errors, label="train")
        plt.semilogx(C_range, valid_errors, label="valid")
        plt.legend();
        plt.xlabel(hyper)
        plt.ylabel("error rate")                
    
    
    def build_classifier_pipeline(self, feature_set, 
                                        classifier = LogisticRegression()):
        '''
        :param feature_set:
        :param classifier:
        :return:
        '''
        print('Classifier: ', classifier)
        print('Feature set: ', feature_set)
        print('COMMENTS COL: ', self.comments_col)
        feats = build_feature_pipelines_and_unions(feature_set, comments_col = self.comments_col)
        pipeline = Pipeline([
            ('features', feats),
            ('classifier', classifier),
        ])
        return pipeline

    
    def train_pipeline(self,
                       save_model_path = '../../models/saved_model.h5',                       
                       feature_set = ['text_feats', 
                                       'length_feats',
                                       'argumentation_feats',
                                       'COMMENTIQ_feats',
                                       'named_entity_feats',
                                       'perspective_content_value_feats',
                                       'perspective_aggressiveness_feats',
                                       'perspecitive_toxicity_feats'],
                       classifier = LogisticRegression()):
        '''
        :return:
        '''
        self.pipeline = self.build_classifier_pipeline(feature_set = feature_set)
        counts_dict = self.y1.value_counts().to_dict()
        counts_dict['Constructive'] = counts_dict.pop(1)
        counts_dict['Non constructive'] = counts_dict.pop(0)        
        print('Size of the training data: ', self.X1.shape[0], 
              '\tConstructive (', counts_dict['Constructive'], ')', 
              '\tNon constructive (', counts_dict['Non constructive'],')'
             )
        
        self.pipeline.fit(self.X_train, self.y_train)              
        joblib.dump(self.pipeline, model_path)
        s = pickle.dumps(self.pipeline)
        print('Model trained and pickled in file: ', model_path)                
        return self.pipeline

def get_arguments():
    parser = argparse.ArgumentParser(description='Constructiveness Feature Extractor')

    parser.add_argument('--feats_csv', '-f', type=str, dest='feats_csv', action='store',
                        default= os.environ['C3_FEATS'], 
                        help="The path for the features CSV.")
            
    args = parser.parse_args()
    return args

def split_data(X, y):
    """
    """
    X_trainvalid, X_test, y_trainvalid, y_test = train_test_split(X, y, train_size=0.8, random_state=1)
    X_train, X_valid, y_train, y_valid = train_test_split(X_trainvalid, y_trainvalid, train_size=0.75, random_state=1)

    print("Number of training examples:", len(y_train))
    print("Number of validation examples:", len(y_valid))
    print("Number of test examples:", len(y_test))
    
    return X_train, y_train, X_valid, y_valid, X_trainvalid, y_trainvalid, X_test, y_test

if __name__=="__main__":
    args = get_arguments()
    df = pd.read_csv(args.feats_csv)
    df['constructive_binary'] = df['constructive'].apply(lambda x: 1 if x > 0.5 else 0)
    feature_set = ['text_feats', 
                   'length_feats',
                   'argumentation_feats',
                   'COMMENTIQ_feats',
                   'named_entity_feats',
                   'perspective_content_value_feats',
                   'perspective_aggressiveness_feats',
                   'perspecitive_toxicity_feats'    
               ]

    feature_cols = [
               #'SEVERE_TOXICITY_probability',
               #'SEXUALLY_EXPLICIT_probability', 'TOXICITY_probability',
               #'TOXICITY_IDENTITY_HATE_probability', 'TOXICITY_INSULT_probability',
               #'TOXICITY_OBSCENE_probability', 'TOXICITY_THREAT_probability',
               #'ATTACK_ON_AUTHOR_probability', 'ATTACK_ON_COMMENTER_probability',
               #'ATTACK_ON_PUBLISHER_probability', 'INCOHERENT_probability',
               #'INFLAMMATORY_probability', 'LIKELY_TO_REJECT_probability',
               #'OBSCENE_probability', 'OFF_TOPIC_probability', 'SPAM_probability',
               #'UNSUBSTANTIAL_probability',
               'has_conjunctions_and_connectives', 'has_stance_adverbials',
               'has_reasoning_verbs', 'has_modals', 'has_shell_nouns', 'length',
               'average_word_length', 'ncaps', 'noov', 'readability_score',
               'personal_exp_score', 'named_entity_count', 'nSents',
               'avg_words_per_sent']    
    X_C3 = df.loc[:, feature_cols]
    y_C3 = df.constructive_binary    
    X_C3_train, y_C3_train, X_C3_valid, y_C3_valid, X_C3_trainvalid, y_C3_trainvalid, X_C3_test, y_C3_test = split_data(X_C3,y_C3)
    classifier = FBConstructivenessClassifier(df, feature_set, X_C3_train, y_C3_train, X_C3_valid, y_C3_valid)
    pipeline = classifier.train_pipeline()
    classifier.show_score(pipeline)
    #classifier.train_classifier(model_path = Config.MODEL_PATH + 'svm_model_automatic_feats.pkl', feature_set = feature_set)
    
