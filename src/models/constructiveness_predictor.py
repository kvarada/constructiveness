__author__ = 'VaradaKolhatkar'

import pandas as pd
import numpy as np
from sklearn.externals import joblib
import sys, os
sys.path.append(os.environ['HOME'] + '/src/models/')
from deeplearning_models import DLTextClassifier

sys.path.append(os.environ['HOME'] + 'src/feature_extraction/')

#import feature_extractor
#from feature_extractor import FeatureExtractor

class ConstructivenessPredictor():
    def __init__(self, 
                 #svm_model_path = ,
                 bilstm_model_path = os.environ['HOME'] + '/models/LSTM_C3_train.h5', 
                 cnn_model_path = os.environ['HOME'] + '/models/CNN_C3_train.h5'):
        '''
        
        Keyword arguments: 
        param model_path: str (model path)

        Description: This class assumes that you have a feature-based trained model. It returns the prediction of the
        given example based on the trained model.
        '''
        # load svm model
        #self.svm_pipeline = joblib.load(svm_model_path)
        
        # load bilstm model
        self.bilstm_classifier = DLTextClassifier(model_path = bilstm_model_path, mode = 'test')    
        
        # load cnn model
        self.cnn_classifier = DLTextClassifier(model_path = cnn_model_path, mode = 'test')

        
    def predict_svm(self, comment):
        """
        """
        pass 
    
    def predict_bilstm(self, comment):
        '''
        '''
        prediction = ""
        prediction_score = self.bilstm_classifier.predict([comment])[0][0]

        if  prediction_score <= 0.5:            
            prediction = 'NON_CONSTRUCTIVE (Score = %.2f)' % prediction_score
        else: 
            prediction = 'CONSTRUCTIVE (Score = %.2f)' % prediction_score
        return prediction
        
    def predict_cnn(self, comment):
        """    
        """    
        prediction = ""
        prediction_score = self.cnn_classifier.predict([comment])[0][0]
        
        if prediction_score <= 0.5:
            prediction = 'NON_CONSTRUCTIVE (Score = %.2f)' % prediction_score
        else: 
            prediction = 'CONSTRUCTIVE (Score = %.2f)' % prediction_score
            
        return prediction
        
if __name__ == "__main__":
    
    #df = pd.read_csv(Config.SOCC_COMMENTS_PATH)
    #svm_predictor = ConstructivenessPredictor()
    #svm_predictor.predict_svm_batch(df)
    #test_set_df = pd.read_csv(Config.TRAIN_PATH + 'SOCC_features.csv')
    #svm_pipeline = joblib.load(Config.MODEL_PATH + 'svm_model_automatic_feats.pkl')
    #test_set_df['constructive_svm_prediction'] = svm_pipeline.predict(test_set_df)
    #test_set_df.rename(columns = {'pp_comment_text':'comment_text'}, inplace = True)
    #cols = ['comment_counter', 'comment_text', 'constructive_svm_prediction']
    #test_set_df.to_csv(Config.RESULTS_PATH + 'SVM_SOCC_predictions.csv', columns = cols, index = False)
    #print('Prediction CSV written: ', Config.RESULTS_PATH + 'SVM_SOCC_predictions.csv')
    #sys.exit(0)
    example1 = r'Allowing mercenaries to run the war is a truly frightening development. Contractors should only be used where the US Army truly lacks resources or expertise. If the Afghan government has any sensible people in charge who care for their country, they should vigorously protest the decision to hand the war effort over to mercenaries. This is a sure way to increase the moral hazards a thousand fold, hide war crimes, and increase corruption beyond even the high levels that exist today.'
    example2 = r'This is rubbish!!!'
    predictor = ConstructivenessPredictor()

    #prediction = predictor.predict_svm(example1)
    #print("Comment: ", example1)
    #print('SVM prediction: ', prediction)

    prediction = predictor.predict_bilstm(example1)
    print("Comment: ", example1)
    print('BILSTM prediction: ', prediction)
    
    prediction = predictor.predict_cnn(example1)
    print("Comment: ", example1)
    print('CNN prediction: ', prediction)
    
    
    #prediction = predictor.predict_svm(example2)
    #print("Comment: ", example2)
    #print('SVM prediction: ', prediction)

    prediction = predictor.predict_bilstm(example2)
    print("Comment: ", example2)
    print('BILSTM prediction: ', prediction)

    prediction = predictor.predict_cnn(example2)
    print("Comment: ", example2)
    print('CNN prediction: ', prediction)



    
    

