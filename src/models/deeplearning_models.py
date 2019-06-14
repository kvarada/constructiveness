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
from keras.layers import Conv1D,Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, MaxPooling1D
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras import regularizers
from keras.layers import Concatenate
from keras import metrics
from sklearn.metrics import f1_score

import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

import pickle, string

class DLTextClassifier():
    def __init__(self, X_train, y_train,  
                 embedding_dimension = 300,
                 max_features = 50000, 
                 maxlen = 100):
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
        self.tokenizer = Tokenizer(num_words=self.max_features)                        
        self.glove_embeddings_dict = defaultdict()        
        #self.read_glove_embeddings()
        #pickle_out = open('../../data/interim/glove_embeddings_dict.pkl',"wb")
        #pickle.dump(self.glove_embeddings_dict, pickle_out) 
        #pickle_out.close()
        pickled_data = open('/home/vkolhatk/dev/constructiveness/data/interim/glove_embeddings_dict.pkl',"rb")
        self.glove_embeddings_dict = pickle.load(pickled_data)
        self.X_train_padded = self.prepare_data(X_train)
        self.embedding_matrix = self.create_embedding_matrix()
        
    def read_glove_embeddings(self):
        """
        """
        gf = open(os.environ['GLOVE_EMBEDDINGS_PATH'], 'r')            
        for line in gf:
            values = line.split()
            word = " ".join(values[0:-self.embedding_dimension])
            try: 
                embedding = np.asarray(values[-self.embedding_dimension:], dtype='float32')                        
                self.glove_embeddings_dict[word] = embedding
            except:
                print(values)
                sys.exit(0)

    def create_embedding_matrix(self):
        """
        """
        not_found = 0 
        self.vocab_size = len(self.word_index) + 1 
        embedding_matrix = np.zeros((self.vocab_size, self.embedding_dimension))

        for word, i in self.word_index.items():
            if word in self.glove_embeddings_dict:
              # words not found in embedding index will be all-zeros.
               embedding_matrix[i] = self.glove_embeddings_dict[word]
            else:
               #print('Not found: ', word)
               not_found += 1

        print('Number of words not found in glove embeddings: ', not_found)
        nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
        print('Percentage non-zero elements: ', nonzero_elements / self.vocab_size)
        return embedding_matrix
    
    
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
         

    def build_bilstm(self):
        """        
        Build an LSTM model self.model using Keras and and print 
        the summary of the model. 
        
        Keyword arguments:
        --------------
        None        
        
        """        
        print('Building model...')
        self.model = Sequential()    
        embedding_layer = Embedding(len(self.word_index) + 1,
                                    self.embedding_dimension,
                                    weights=[self.embedding_matrix],
                                    input_length=self.maxlen,
                                    trainable=False)
        self.model.add(embedding_layer)        
        #self.model.add(Embedding(self.max_features, self.embedding_dimension))
        self.model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
        self.model.add(Dropout(0.5))
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
        self.model.add(Embedding(self.vocab_size, 
                                self.embedding_dimension, 
                                weights=[self.embedding_matrix], 
                                input_length=self.maxlen, 
                                trainable=True))        
        
        self.model.add(Dropout(0.5))
        # we add a Convolution1D, which will learn filters
        # word group filters of size filter_length:
        self.model.add(Conv1D(self.filters,
                             self.kernel_size,
                             padding='valid',
                             activation='relu',
                             strides=1,
                             kernel_regularizer=regularizers.l2(0.01)))
        self.model.add(MaxPooling1D())
        self.model.add(Conv1D(self.filters,
                             self.kernel_size,
                             padding='valid',
                             activation='relu',
                             strides=1,
                             kernel_regularizer=regularizers.l2(0.01)))
        self.model.add(MaxPooling1D())
        self.model.add(Conv1D(self.filters,
                             self.kernel_size,
                             padding='valid',
                             activation='relu',
                             strides=1,
                             kernel_regularizer=regularizers.l2(0.01)))

        
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
              save_path = os.environ['HOME'] + '/models/my_model.h5'):
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
        
        self.model.fit(self.X_train_padded, y_train,
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
        predictions = self.model.predict_classes(X_test_padded)    
        print('sklearn micro-F1-Score:', f1_score(y_test, predictions, average='micro'))
        #score, acc = self.model.evaluate(X_test_padded, y_test)        
        #print('mae: ', mae)
        #print('acc: ', acc)        

    def write_model_scores_df(self, C3_test_df, 
                              results_csv_path = os.environ['HOME'] + 'models/CNN_C3_test_predictions1.csv'):
        """
        Write the model scores as a CSV in the provided X_test and y_test. 
        
        Keyword arguments:
        --------------
        X_test -- (list) the X values (encoded and padded list documents)
        y_test -- (list) the y values (a list of labels)
        
        """
        
        results_df = C3_test_df
        X_test_padded = self.prepare_data(results_df['pp_comment_text'].astype(str), mode='test')        
        results_df['prediction'] = self.model.predict_classes(X_test_padded)    
        results_df['prediction proba'] = self.model.predict(X_test_padded)
        results_df['comment_len'] = results_df['pp_comment_text'].apply(lambda x: len(str(x).split()))
        results_df.to_csv(results_csv_path, index = False)
        print('Predictions file written: ', results_csv_path)

                
    def predict(self, texts):
        """
        Predict the labels of the given texts using self.model. 
        
        Arguments:
        --------------
        texts -- (list) a list of texts for prediction. 
        
        Returns: 
        --------------
        The predictions correponding to the provided texts. 
        
        """
        padded_sequences = self.prepare_data(texts, mode = 'test')
        self.model.predict(padded_sequences)    
       
    
def run_dl_experiment(C3_train_df, 
                      C3_test_df, 
                      model = 'cnn'):


    """    
    """    
    X_train = C3_train_df['pp_comment_text'].astype(str)
    y_train = C3_train_df['constructive_binary']
    
    X_test = C3_test_df['pp_comment_text'].astype(str)
    y_test = C3_test_df['constructive_binary']
    
    dlclf = DLTextClassifier(X_train, y_train)
    
    if model.endswith('lstm'):
        dlclf.build_bilstm()
        
    elif model.endswith('cnn'): 
        dlclf.build_cnn()
        
    dlclf.train(X_train, y_train)
    print('\n Train accuracy: \n\n')
    dlclf.evaluate(X_train, y_train)
    
    print('\n Test accuracy: \n\n')
    dlclf.evaluate(X_test, y_test)
    results_df = dlclf.write_model_scores_df(C3_test_df)
       
        
if __name__=="__main__":    

    # Run DL experiments on length-balanced C3
   
    #C3_train_df = pd.read_csv(os.environ['C3_MINUS_LB'])
    #C3_test_df = pd.read_csv(os.environ['C3_LB'])
    C3_train_df = pd.read_csv(os.environ['C3_TRAIN'])
    C3_test_df = pd.read_csv(os.environ['C3_TEST'])    
       
    #X_train = C3_train_df['pp_comment_text'].astype(str)
    #y_train = C3_train_df['constructive_binary']
    
    #X_test = C3_test_df['pp_comment_text'].astype(str)
    #y_test = C3_test_df['constructive_binary']
    #print('CNN experiment on the length-balanced test set: ')
    run_dl_experiment(C3_train_df, C3_test_df, model = 'cnn')
    
    sys.exit(0)    
    # Run DL experiments on C3
    C3_train_df = pd.read_csv(os.environ['C3_TRAIN'])
    C3_test_df = pd.read_csv(os.environ['C3_TEST'])    
    
    X_train = C3_train_df['pp_comment_text']
    y_train = C3_train_df['constructive_binary']
    
    X_test = C3_test_df['pp_comment_text']
    y_test = C3_test_df['constructive_binary']
    print('CNN experiment on the C3 test set:')
    run_dl_experiment(X_train, y_train, X_test, y_test, model='cnn')
    
    