# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 19:14:58 2023

@author: hp
"""

import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer


import tensorflow as tf
from googletrans import Translator
def trans(x):
   if x not in ['user_type','description','region_name','city_name','parent_category','category','p1','p2','title']:
     translator=Translator(service_urls=['translate.googleapis.com'])
     s=translator.translate(x, dest='en')
     c=s.text
     return c
   else:
       return 'enter a valid value'
#print(trans("Резиновые сапоги Lucky Land"))

#data preprocessing

lemmatizer = WordNetLemmatizer()
sw_nltk = stopwords.words('english')


def prep(x):
  b=re.sub(r'[^a-zA-Z]',' ',x)                           
  b2 = [lemmatizer.lemmatize(word) for word in b.split() if word.lower() not in sw_nltk]   
  new_text = " ".join(b2) 
  return new_text
#print(prep("Selling my beloved car!! Fast, bright, stylish."))


import tensorflow as tf
import pickle
from keras.preprocessing.text import Tokenizer  
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokk_re=pickle.load(open('C:/Users/hp/Pictures/web_app2/tok_pc.sav','rb'))
tt=tokk_re.texts_to_sequences(["Real estate"])
maxlen_re=4
train_padded_re = pad_sequences(tt, padding= "post", truncating="post", maxlen=maxlen_re)
print(train_padded_re)



import cv2
import numpy as np


##preprocessing image data
import tensorflow as tf
def img_pre(df):
  height = 128
  width = 128
  dim = (width, height)
  c=0
  x=[]
  z=[]  
  p='C:/Users/hp/Pictures/web_app2/' + df                                                       ##using "cv2.imread" we could see some are None .so this None is seen only after reading an image, so inoreder to remove such image file we append in a list 'None' and rest of the image stored in same list.so when we form dataframe using such image file we could drop such rows which is None
  img=cv2.imread(p, cv2.IMREAD_UNCHANGED)
  print(img)
  res = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
  image = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
  image=image/255
  x.append(image) 
  x1=np.array(x)
  print(x1.shape)
  return x1
dd1=img_pre('0a0c6b44850953b409b483423c867ac9bef890f48594c50434e99292fc405239.jpg')

import re
def act_d(x):
     if x!='activation_date':
       f1=[]
       f2=[]
       f3=[]
       b=re.split(r'-',x)
       f1.append(int(b[0]))
       f2.append(int(b[1]))
       f3.append(int(b[2]))
       return f1,f2,f3
     else:
        return 'enter a valid value'
f1,f2,f3= act_d('2-3-2017')
print(f1)
print(f2)
    
