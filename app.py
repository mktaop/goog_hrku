#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 15:45:08 2022

@author: avi_patel
"""
#pip install flask
from flask import Flask,render_template,request
import numpy as np
#import keras
#import tensorflow 
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

model=load_model('/Users/avi_patel/Downloads/googlstm.h5')
 
app = Flask(__name__)
 
#@app.route('/',methods=['POST','GET'])
@app.route('/')
def new():
    return render_template('new.html')
 
@app.route('/predict', methods=['POST'] )
def predict():
    # getting data
    data1=float(request.form['a'])
    data2=float(request.form['b'])
    data3=float(request.form['c'])
    data4=float(request.form['d'])
    data5=float(request.form['e'])
 
    # preparing for the prediction
    features=np.array([data1,data2,data3,data4,data5])
    d2 = np.reshape(features, (-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    d2f = scaler.fit_transform(d2)
    d2x= np.reshape(d2f, (d2f.shape[1], 1, d2f.shape[0]))
    d2xhat=model.predict(d2x)
    prediction=scaler.inverse_transform(d2xhat)
    #pred = model.predict([features])
    statement='Prediciton for the next day is: ' + prediction 

    
    #def statement():
        #if pred == 0:
           # return 'Result:- The model has predicted that you will not suffer from any cardic arresst but you should take care of your self.'
       # else:
           # return 'Result:- You should consult with doctor, The model has predicted that you will suffer form cardic arrest.'
    
    return render_template('new.html',statement=statement())
 
if __name__=='__main__':
    app.run(debug=True)
