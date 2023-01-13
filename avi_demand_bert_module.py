# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 12:17:50 2023

@author: hp
"""
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
from tensorflow.keras.models import Model
from transformers import BertTokenizer, BertModel                               #tokenize the text data   
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  


tf.keras.backend.clear_session()

# maximum length of a seq in the data we have, for now i am making it as 55. You can change this
max_seq_length = 55

#BERT takes 3 inputs

#this is input words. Sequence of words represented as integers
input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")

#mask vector if you are padding anything
input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask")

#segment vectors. If you are giving only one sentence for the classification, total seg vector is 0. 
#If you are giving two sentenced with [sep] token separated, first seq segment vectors are zeros and 
#second seq segment vector are 1's
segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="segment_ids")

#bert layer 
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=False)
pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

#Bert model
#We are using only pooled output not sequence out. 
#If you want to know about those, please read https://www.kaggle.com/questions-and-answers/86510
bert_model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=sequence_output)


from tensorflow.keras.preprocessing.sequence import pad_sequences
a1=[]
a2=[]
a3=[]

def ber_pre(x):
   tok=tokenizer.tokenize(x)                                 #tokenization of each text sentences in xtrain
   max_len=55
   if len(tok)>=53:
      tok=tok[0:(max_len-2)]                                                    #when the length of token is greater than or equal to 55 that text sentence is truncated
      tok.insert(0,'[CLS]')                                                     #https://stackoverflow.com/questions/67594924/adding-start-and-end-tokens-to-lines-of-a-tokenized-document
      tok.insert(len(tok),'[SEP]')                                              #adding [CLS] at the start  and [SEP] at the end of token
      mask=np.array([1]*(len(tok))+[0]*(max_len-len(tok)))                      #massk the tokens
      a2.append(mask)                                                           #storing it in a list a2
    
   if len(tok)<53:                                                              #if it is less than 55, add '[PAD]' 
     c=53-len(tok)                        
     for i in range(c):
       tok.insert(len(tok)+1,'[PAD]')                                           #[PAD] is added in order to make same length text token sentence
     tok.insert(0,'[CLS]')                                                      #https://stackoverflow.com/questions/67594924/adding-start-and-end-tokens-to-lines-of-a-tokenized-document
     tok.insert(len(tok),'[SEP]')                                               #[CLS] is added at the start and [SEP] at the end of tokens
     mask=np.array([1]*(len(tok)-c)+[0]*(max_len-(len(tok)-c)))                 #mask the token sentences
     a2.append(mask)                                                            #storing it in a list a2
   token_id=tokenizer.convert_tokens_to_ids(tok)                                #positional encoding the tokens in oder to get position of every tokens
   a1.append(token_id)                                                          #storing in a list a1
   #the segment array
   segment=np.array([0]*max_len)                                                #Create a segment input for train. We are using only one sentence so all zeros
   a3.append(segment) 
   X_train_tokens_des=np.array(a1)
   X_train_mask_des=np.array(a2)
   X_train_segment_des=np.array(a3)    
   X_train_pooled_output_des=bert_model.predict([X_train_tokens_des,X_train_mask_des,X_train_segment_des])

   return  X_train_pooled_output_des                                                       #storing in a list a3


