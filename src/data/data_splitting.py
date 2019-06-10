import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score

class DataSplitter:
    
    def __init__(self, df,
                       text_col = 'pp_comment_text',
                       label_col = 'constructive_binary'):
        """
        """
        self.text_col = text_col
        self.label_col = label_col
        self.df = df
        self.X = C3_feats_df[text_col]
        self.y = C3_feats_df[label_col]
        
            
    def create_length_balanced_splits(self,
                                      lower_upper_ranges = [[10,20], [20, 30], [30, 40], [40, 50], [50, 60], [60, 70], [70, 80]]     ):
        """
        """
        comment_len_col = 'comment_len'
        print(self.df[self.text_col].dtype)
        #self.df[self.text_col + 'len'] = self.df[self.text_col].str.split()
        self.df[comment_len_col] = self.df[self.text_col].apply(lambda x: len(str(x).split()))

        balanced_dfs = []
        for (lower, upper) in lower_upper_ranges:
            print('LOWER: ', lower)
            print('UPPER: ', upper)    
            subset_df = self.df[(self.df[comment_len_col] >= lower) & (self.df[comment_len_col] < upper)]
            d = subset_df[self.label_col].value_counts().to_dict()
            lower_count = min(list(d.values()))
            con_subset_df = subset_df[subset_df[self.label_col] == 1.0].sample(n=lower_count)
            print('Number of constructive samples: ', con_subset_df.shape[0])
            non_con_subset_df = subset_df[subset_df[self.label_col] == 0.0].sample(n=lower_count)
            print('Number of non-constructive samples: ', non_con_subset_df.shape[0])
            balanced_dfs.extend([con_subset_df, non_con_subset_df])
            
        self.df_lb = pd.concat(balanced_dfs)    

        self.df_minus_lb = self.df[~self.df['comment_counter'].isin(self.df_lb['comment_counter'].tolist())]
        
        self.X_train_lb = self.df_minus_lb[[self.text_col, 'comment_text']]   
        self.y_train_lb = self.df_minus_lb[self.label_col]

        self.X_test_lb = self.df_lb[[self.text_col, 'comment_text']]
        self.y_test_lb = self.df_lb[self.label_col]
        print('X_train len: ', len(self.X_train_lb))
        print('X_test len: ', len(self.X_test_lb))

        # Write train/test CSVs for length-balanaced experiments 
        train_df = pd.concat([self.X_train_lb, self.y_train_lb], axis=1)
        train_df.to_csv(os.environ['C3_MINUS_LB'], index = False)
        print('Length-balanced train CSV file written: ', os.environ['C3_MINUS_LB'])

        test_df = pd.concat([self.X_test_lb, self.y_test_lb], axis=1)    
        test_df.to_csv(os.environ['C3_LB'], index = False)
        print('Length-balanced test CSV file written: ', os.environ['C3_LB'])    
        
        return self.X_train_lb, self.y_train_lb, self.X_test_lb, self.y_test_lb        
        

    def get_length_balanced_data_splits(self):
        """
        """
        
        return self.X_train_lb, self.y_train_lb, self.X_test_lb, self.y_test_lb
    
    
    def create_C3_data_splits(self, train_size = 0.8):
        """
        """
        self.X_trainvalid, self.X_test, self.y_trainvalid, self.y_test = train_test_split(self.X, self.y, train_size = train_size, random_state=1)
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X_trainvalid, self.y_trainvalid, train_size = train_size, random_state=1)

        print("Number of training examples:", len(self.y_train))
        print("Number of validation examples:", len(self.y_valid))
        print("Number of test examples:", len(self.y_test))
        # Write train/test CSVs for length-balanaced experiments 
        train_df = pd.concat([self.X_trainvalid, self.y_trainvalid], axis=1)
        train_df.to_csv(os.environ['C3_TRAIN'], index = False)
        print('C3 train split CSV file written: ', os.environ['C3_TRAIN'])

        test_df = pd.concat([self.X_test, self.y_test], axis=1)    
        test_df.to_csv(os.environ['C3_TEST'], index = False)
        print('C3 test CSV file written: ', os.environ['C3_TEST'])
        
    def get_C3_data_splits(self): 
        return self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_trainvalid, self.y_trainvalid, self.X_test, self.y_test

    
if __name__=="__main__":
    C3_feats_df = pd.read_csv(os.environ['C3_FEATS_PREPROCESSED'])    
    #C3_feats_df = C3_feats_df.drop(C3_feats_df[ (C3_feats_df['constructive'] <= 0.6) & (C3_feats_df['constructive'] >= 0.4)].index)
    print(C3_feats_df.shape)
    C3_feats_df['constructive_binary'] = C3_feats_df['constructive'].apply(lambda x: 1.0 if x > 0.5 else 0)            
    ds = DataSplitter(C3_feats_df)
    ds.create_length_balanced_splits()
    ds.create_C3_data_splits()
    
    
    
 
    
    
    
    
