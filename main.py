import streamlit as st
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
import pickle
import string
from nltk.corpus import stopwords
import nltk

def transform_text(text):
    #convert to lower case
    text = text.lower()
    #tokenize the text
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]#text = y is wrong since strings are mutable in python
    y.clear()
    #removing stop words
    for i in text:#[c for c in text if c not in string.punctuns]
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    #stem the text
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")
if st.button('Predict'):
 #1.preprocess
 transform_sms=transform_text(input_sms)
 #2.vectorize
 vector = tfidf.transform([transform_sms])
 #3.predict
 result = model.predict(vector)
 #4.display
 if result ==0 :
  st.header("NOT SPAM")
  # print("not spam")
 else:
  st.header("SPAM")
  # print("spam")

