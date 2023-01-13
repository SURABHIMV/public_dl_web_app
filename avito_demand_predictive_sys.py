# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 07:51:28 2023

@author: hp
"""
import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle
from keras.preprocessing.text import Tokenizer  
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, model_from_json  
from tensorflow import keras 
import numpy as np
import avito_demand_module
import avi_demand_bert_module


#loading the saved tokenizers,models

tokk_re=pickle.load(open('C:/Users/hp/Pictures/web_app2/tok_re.sav','rb'))
tokk_pc=pickle.load(open('C:/Users/hp/Pictures/web_app2/tok_pc.sav','rb'))
tokk_c=pickle.load(open('C:/Users/hp/Pictures/web_app2/tok_c.sav','rb'))
tokk_ci=pickle.load(open('C:/Users/hp/Pictures/web_app2/tok_ci.sav','rb'))
tokk_p1=pickle.load(open('C:/Users/hp/Pictures/web_app2/tok_p1.sav','rb'))
tokk_p2=pickle.load(open('C:/Users/hp/Pictures/web_app2/tok_p2.sav','rb'))
tokk_u=pickle.load(open('C:/Users/hp/Pictures/web_app2/tok_u.sav','rb'))

stdd_p=pickle.load(open('C:/Users/hp/Pictures/web_app2/std_p.sav','rb'))
stdd_it=pickle.load(open('C:/Users/hp/Pictures/web_app2/std_it.sav','rb'))
stdd_im=pickle.load(open('C:/Users/hp/Pictures/web_app2/std_im.sav','rb'))
stdd_y=pickle.load(open('C:/Users/hp/Pictures/web_app2/std_y.sav','rb'))
stdd_m=pickle.load(open('C:/Users/hp/Pictures/web_app2/std_m.sav','rb'))
stdd_d=pickle.load(open('C:/Users/hp/Pictures/web_app2/std_d.sav','rb'))

#loading the saved trained model
json_file = open('C:/Users/hp/Pictures/web_app2/avi_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights('C:/Users/hp/Pictures/web_app2/avi_weightx.h5')
print("Loaded model from disk")


def avito_demand_prediction(price1,item_seq_number1,user_type1,image_top_1_1,image1,description_en_t1,region_en_t1,city_en_t1,parent_category_name_en_t1,category_name_en_t1,param_1_en_t1,param_2_en_t1,title_en_t1,activation_day1):
    #preprocessing the categorical data
    t1=avito_demand_module.trans(user_type1)
    if (t1=='enter a valid value'):
     return 'enter a valid value'
    p1=avito_demand_module.prep(t1)
    tok1=tokk_u.texts_to_sequences(p1)
    maxlen_u=1
    train_padded_u = pad_sequences(tok1, padding= "post", truncating="post", maxlen=maxlen_u)
    
    t2=avito_demand_module.trans(description_en_t1)
    if (t2=='enter a valid value'):
     return 'enter a valid value'
    p2=avito_demand_module.prep(t2)
    tok_bert_des=avi_demand_bert_module.ber_pre(p2)

    t3=avito_demand_module.trans(region_en_t1)
    if (t3=='enter a valid value'):
     return 'enter a valid value'
    p3=avito_demand_module.prep(t3)
    tok3=tokk_re.texts_to_sequences(p3)
    maxlen_re=4
    train_padded_re = pad_sequences(tok3, padding= "post", truncating="post", maxlen=maxlen_re)
    
    t4=avito_demand_module.trans(city_en_t1)
    if (t4=='enter a valid value'):
     return 'enter a valid value'
    p4=avito_demand_module.prep(t4)
    tok4=tokk_ci.texts_to_sequences(p4)
    maxlen_ci=3
    train_padded_ci = pad_sequences(tok4, padding= "post", truncating="post", maxlen=maxlen_ci)
    
    t5=avito_demand_module.trans(parent_category_name_en_t1)
    if (t5=='enter a valid value'):
     return 'enter a valid value'
    p5=avito_demand_module.prep(t5)
    tok5=tokk_pc.texts_to_sequences(p5)
    maxlen_pc=2
    train_padded_pc = pad_sequences(tok5, padding= "post", truncating="post", maxlen=maxlen_pc)
    
    t6=avito_demand_module.trans(category_name_en_t1)
    if (t6=='enter a valid value'):
     return 'enter a valid value'
    p6=avito_demand_module.prep(t6)
    tok6=tokk_c.texts_to_sequences(p6)
    maxlen_c=2
    train_padded_c = pad_sequences(tok6, padding= "post", truncating="post", maxlen=maxlen_c)
    
    t7=avito_demand_module.trans(param_1_en_t1)
    if (t7=='enter a valid value'):
     return 'enter a valid value'
    p7=avito_demand_module.prep(t7)
    tok7=tokk_p1.texts_to_sequences(p7)
    maxlen_p1=2
    train_padded_p1 = pad_sequences(tok7, padding= "post", truncating="post", maxlen=maxlen_p1)
    
    t8=avito_demand_module.trans(param_2_en_t1)
    if (t8=='enter a valid value'):
     return 'enter a valid value'
    p8=avito_demand_module.prep(t8)
    tok8=tokk_p2.texts_to_sequences(p8)
    maxlen_p2=2
    train_padded_p2 = pad_sequences(tok8, padding= "post", truncating="post", maxlen=maxlen_p2)
    
    t9=avito_demand_module.trans(title_en_t1)
    if (t9=='enter a valid value'):
     return 'enter a valid value'
    p9=avito_demand_module.prep(t9)
    tok_bert_t=avi_demand_bert_module.ber_pre(p9)
    
    #preprocessing the image data
    if len(image1)!='image_name':
      m=avito_demand_module.img_pre(image1)
    else: 
        return 'enter a valid value'
        
        
    #numerical data preprocessing
    if price1==0:
        return "enter a valid value"
    q1=np.array(price1)
    s1=stdd_p.transform(q1.reshape(1,-1))
    
    if item_seq_number1==0:
        return 'enter a valid value'
    q2=np.array(item_seq_number1)
    s2=stdd_it.transform(q2.reshape(1,-1))
    
    if image_top_1_1==0:
        return 'enter a valid value'
    q3=np.array(image_top_1_1)
    s3=stdd_im.transform(q3.reshape(1,-1))
    
    n1,n2,n3=avito_demand_module.act_d(activation_day1)
    q4=np.array(n1)
    s4=stdd_y.transform(q4.reshape(1,-1))
    q5=np.array(n2)
    s5=stdd_m.transform(q5.reshape(1,-1))
    q6=np.array(n3)
    s6=stdd_d.transform(q6.reshape(1,-1))

    num=np.hstack([s1,s2,s3,s4,s5,s6])
    
    mod=loaded_model.predict([tok_bert_des[:1],train_padded_re[:1],train_padded_ci[:1],train_padded_pc[:1],train_padded_c[:1],tok_bert_t[:1],train_padded_p1[:1],train_padded_p2[:1],train_padded_u[:1],m[:1],num[:1]])
    print(mod[0])
    if mod[0]<0.5:
        return 'no demand'
    else:
        return 'demand'

#streamlit part were we use user interface
def main():
    #giving a title
    st.title('Avito Demand Prediction web App')
    
    #load 1000 rows of data into the dataframed
    #st.write(data)
    

    #getting the input from user
    
   
   
    #price=st.number_input(label='price')
    price=st.selectbox("price of the product",[0,500,18000,4000,500,400])
   
    
    #item_seq_number=st.number_input(label='item_seq_number')
    item_seq_number=st.selectbox("seq number of an item",[0,61,6,80,13,2062])
    
    
   
   
    #user_type=st.text_input(label='user_type',type='password')
    user_type=st.selectbox("type of user",['user_type','private','private','Company','private','company'])
   
    
    #image_top_1=st.number_input(label='image_top_1')
    image_top_1=st.selectbox("image_top_1",[0,567,1396,444,42,632])
    
    #image=st.text_input(label='image')
    image=st.selectbox("image of ad",['image_name','9bab29a519e81c14f4582024adfebd4f11a4ac71d323a62f7731e19db9702115.jpg','645d3fb949cb116a00c596ca1e168d8c5ddd21cdeacc5a26b711e3921a06f536.jpg','9adfe931504bcd338463b40f98c2f28aceaf0acae787a17700f8bf6215115fb5.jpg','d1a73a4bf5bb768660deebfb8c672ecaae4863d45c39a138282ab29496aa7307.jpg','c8b120be7d1c66cb99d17b111f352c8160ef512e5cb49611eb506ea2b0e6928a.jpg'])
   
    #description_en_t=st.text_input(label='description_en_t',type='password')
    description_en_t=st.selectbox("description of ad",['description','Бойфренды в хорошем состоянии.','Сдается однокомнатная мебелированная квартира квартира. Ежемесячная плата 18 тыс.р. + свет.','БЕРЦЫ BATES Gore-Tex®, новые. Может надетые пару раз,не больше. Длина стельки - 26,5 см.','Продам. Новый. Нейтральный цвет. Подойдёт как для девочки, так и для мальчика. Написано 12 мес. Маломерит. Мы быстро выросли и не успели одеть.','В хорошем состоянии футболка Adidas, оригинал, про-во Тайланд, чистая, немного б\у, подойдет на размер 44-46'])
    #region_en_t=st.text_input(label='region_en_t',type='password')
    region_en_t=st.selectbox("region",['region_name','Пермский край','Ханты-Мансийский АО','Владимирская область','Ханты-Мансийский АО','Волгоградская область'])
    
    #city_en_t=st.text_input(label='city_en_t',type='password')
    city_en_t=st.selectbox("city",['city_name','Пермь','Ханты-Мансийск','Владимир','Пойковский','Волгоград'])
    
    #parent_category_name_en_t=st.text_input(label='parent_category_name_en_t',type='password')
    parent_category_name_en_t=st.selectbox('parent_category_name',['parent_category','Личные вещи','Недвижимость','Личные вещи','Личные вещи','Личные вещи'])
    
    #category_name_en_t=st.text_input(label='category_name_en_t',type='password')
    category_name_en_t=st.selectbox("category_name",['category','Одежда, обувь, аксессуары','Квартиры','Одежда, обувь, аксессуары','Детская одежда и обувь','Одежда, обувь, аксессуары'])
    #param_1_en_t=st.text_input(label='param_1_en_t',type='password')
    param_1_en_t=st.selectbox("param_1",['p1','Женская одежда','Сдам','Мужская одежда','Для мальчиков','Мужская одежда'])
    
    #param_2_en_t=st.text_input(label='param_2_en_t',type='password')
    param_2_en_t=st.selectbox("param_2",['p2','Джинсы','На длительный срок','Обувь','Комбинезоны и боди','Трикотаж и футболки'])
    
    #title_en_t=st.text_input(label='title_en_t',type='password')
    title_en_t=st.selectbox("title of ad",['title','Бойфренды colins','1-к квартира, 25 м², 2/2 эт.','Берцы bates Gore-Tex 7R','Летний комбинезончик. новый','Футболка Adidas (хорошее состояние)'])
    
    #activation_day=st.text_input(label='activation_day',type='password')
    activation_day=st.selectbox("activation_day",['activation_date','2017-03-25','2017-03-25','2017-03-24','2017-03-28','2017-03-15'])
    
    
    #prediction code
    avito_demand=''
    
    #creating a button for prediction
    
    if st.button('avito demand prediction'):
       
        avito_demand= avito_demand_prediction(price,item_seq_number,user_type,image_top_1,image,description_en_t,region_en_t,city_en_t,parent_category_name_en_t,category_name_en_t,param_1_en_t,param_2_en_t,title_en_t,activation_day)
       
    st.success(avito_demand)


if __name__=='__main__':
    main()




