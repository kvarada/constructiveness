import pandas as pd
import numpy as np
import numpy.random as npr
import os, sys, re
import collections
from collections import defaultdict

import matplotlib.pyplot as plt

import tensorflow
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU
from keras.layers import Conv1D,Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split


class DLTextClassifier():
    def __init__(self, 
                 embedding_dimension = 200,
                 max_features = 20000, 
                 maxlen = 80):
        """
        Instantiate the DLTextClassifer.
        
        
        Keyword arguments: 
        --------------
        embedding_dimension -- (int) size of the embedding vector
        max_features -- (int) maximum number of features (words) to keep 
                        in the vocabulary
        maxlen -- (int) maximum length of all sequences
        """        
        
        self.embedding_dimension = embedding_dimension
        self.max_features = max_features
        
        # cut texts after this number of words 
        # (among top max_features most common words)
        self.maxlen = maxlen
        self.filters = 250
        self.kernel_size = 3
        self.hidden_dims = 250
        # create the tokenizer        
        self.tokenizer = Tokenizer(num_words=self.max_features, 
                             filters='! #$% ()*+,-./:; = ?@[\\]^_`{|}~\t\n>"<')                        
          
    def prepare_data(self, corpus, mode = 'train'):
        """
        Given a corpus and the mode, prepare data for the deep learning model.
        Maps a sequence words to integers and returns a padded and 
        
        Keyword arguments:
        --------------
        corpus -- (list) a list of documents or texts
        mode   -- (str) a string indicating whether the corpus is the train
                or test corpus. 
        
        Returns:
        --------
        self.pad_sequences -- (list) a list of encoded and padded corpus
        """
        
        if mode == 'train': 
            # fit the tokenizer on the documents
            self.tokenizer.fit_on_texts(corpus)

        # Store word_index
        self.word_index = self.tokenizer.word_index
                
        # integer encode documents
        encoded_corpus = self.tokenizer.texts_to_sequences(corpus)        
        print('len of encoded docs: ', len(encoded_corpus))
        return self.pad_sequences(encoded_corpus)

      
    def pad_sequences(self, data):
        """
        Given data this method returns the padded sequences so that all 
        sequences are of the same length. 
        
        Keyword arguments:
        --------------
        data -- (list) a list of documents or texts

        Returns:
        --------
        a list with encoded integers with shape (len(sequences), maxlen)
        
        """
        
        print('Pad sequences (samples x time)')
        padded_data = sequence.pad_sequences(data, maxlen=self.maxlen)
        print('Padded data shape:', padded_data.shape)
        return padded_data    
      
    def build_lstm(self):
        """        
        Build an LSTM model self.model using Keras and and print 
        the summary of the model. 
        
        Keyword arguments:
        --------------
        None        
        
        """
        
        print('Building model...')
        self.model = Sequential()                               
        self.model.add(Embedding(self.max_features, self.embedding_dimension))
        self.model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(1, activation='sigmoid'))
        # try using different optimizers and different optimizer configs
        self.model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        
        print(self.model.summary())
                

    def build_cnn(self):
        """        
        Build an LSTM model self.model using Keras and and print 
        the summary of the model. 
        
        Keyword arguments:
        --------------
        None        
        
        """        
        print('Building CNN model...')
        self.model = Sequential()                                              
        self.model.add(Embedding(self.max_features, self.embedding_dimension))        
        self.model.add(Dropout(0.2))
        # we add a Convolution1D, which will learn filters
        # word group filters of size filter_length:
        self.model.add(Conv1D(self.filters,
                             self.kernel_size,
                             padding='valid',
                             activation='relu',
                             strides=1))
        # we use max pooling:
        self.model.add(GlobalMaxPooling1D())

        # We add a vanilla hidden layer:
        self.model.add(Dense(self.hidden_dims))
        self.model.add(Dropout(0.2))
        self.model.add(Activation('relu'))

        # We project onto a single unit output layer, and squash it with a sigmoid:
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])        
        print(self.model.summary())
        
        
    def train(self, 
              X_train, y_train,
              batch_size =32, 
              epochs = 5, 
              save_path='../../models/my_model.h5'):
        """
        Given the parameters train a deep learning model and save and return it.  
        
        Keyword arguments: 
        -------------------------
        X_train -- (list) the X values of the train split (a list documents)
        y_train -- (list) the y values of the train split (a list of labels)
        batch_size -- (int) the batch_size for the training
        epochs -- (int) the number of epochs for training 
        save_path -- (str) the path to save the model
        
        Returns: 
        --------------------
        self.model -- the trained model
        """        

        print('Training...')        
        X_train_padded = self.prepare_data(X_train)        
        self.model.fit(X_train_padded, y_train,
                       batch_size=batch_size,
                       validation_split = 0.1,     
                       epochs=epochs)
        self.model.save(save_path) 
        return self.model


    def evaluate(self, X_test, y_test):
        """
        Evaluate self.model on the provided X_test and y_test. 
        
        Keyword arguments:
        --------------
        X_test -- (list) the X values (encoded and padded list documents)
        y_test -- (list) the y values (a list of labels)
        
        """
        
        X_test_padded = self.prepare_data(X_test, mode='test')        
        score, acc = self.model.evaluate(X_test_padded, y_test)
        print('Accuracy: ', acc)
        X_test_padded = self.prepare_data(X_test, mode='test')        
        score, acc = self.model.evaluate(X_test_padded, y_test)
        print('Accuracy: ', acc)

    def predict(self, texts):
        """
        Predict the labels of the given texts using self.model. 
        
        Keyword arguments:
        --------------
        texts -- (list) a list of texts for prediction. 
        
        Returns: 
        --------------
        The predictions correponding to the provided texts. 
        
        """
        padded_sequences = self.prepare_data(texts, mode = 'test')
        return self.model.predict(padded_sequences)
    

if __name__=="__main__":    
    print(os.environ['RAW'])
    #df=pd.read_csv('/home/vkolhatk/dev/constructiveness/data/raw/constructiveness_and_toxicity_annotations_batches_1_to_12.csv')
    #df['constructive_binary'] = df['constructive'].apply(lambda x: 1.0 if x > 0.5 else 0.0)
    #X_train, X_test, y_train, y_test = train_test_split(df['comment_text'], 
    #                                                    df['constructive_binary'], 
    #                                                    test_size=0.2, 
    #                                                    random_state=42)        

    #lstm = DLTextClassifier()
    #lstm.build_lstm()
    #lstm.train(X_train, y_train)
