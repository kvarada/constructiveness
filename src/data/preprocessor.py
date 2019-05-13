import nltk
from nltk.tokenize import sent_tokenize, word_tokenize




if __name__=="__main__":
    df = pd.read_csv('')
    df['constructive_binary'] = df['constructive'].apply(lambda x: 1.0 if x > 0.5 else 0.0)
    