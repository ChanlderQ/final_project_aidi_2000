import pandas as pd
import numpy as np
from bs4 import BeautifulSoup 
import re
import pickle
import nltk
nltk.download('omw-1.4')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



def cleaned_review(review):
    my_stopwords = stopwords.words("english")
    my_stopwords.remove('not')
    my_stopwords.remove('no')
    
    lemmatizer = WordNetLemmatizer()
    if isinstance(review,str):
        # remove any html tags
        new_review = BeautifulSoup(review).get_text()
        
        # remove urls from reviews
        no_urls = new_review.replace('http\S+', '').replace('www\S+', '')
        
        # remove any non-letters
        clean_review = re.sub("[^a-zA-Z]", " ", no_urls)
        
        # convert whole sentence to lowercase and split
        new_words = clean_review.lower().split()
        
        # converting stopwords list to set for faster search
        stops = set(my_stopwords)
        
        # using stopwords to remove irrelavent words and lemmatizing the final output
        final_words = [lemmatizer.lemmatize(word) for word in new_words if not word in stops]
        # return the final result
        return (" ".join(final_words))
    else:
        cleaned_review(str(review))
        
def find_sentiment(review,max_len=40):
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    loaded_model = pickle.load(open("my_saved_model.h5", 'rb'))
    
    model_input=cleaned_review(review)
    seq = tokenizer.texts_to_sequences([model_input])
    padded = pad_sequences(seq, maxlen=max_len)
    pred = loaded_model.predict(padded)
    label = ['Positive','Negative','Neutral']
    return label[np.argmax(pred)]

# test
#input_string="hello, where are you from"
#result=find_sentiment(input_string)
#print(input_string+"  ---The sentiment prediction: "+result)