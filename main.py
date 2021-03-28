# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 09:01:12 2021

@author: kalas
"""

from flask import Flask, make_response
from flask import json
from flask import request
from flask import render_template
import os
import numpy as np
import pandas as pd
import csv
import librosa
import pickle


model1 = pickle.load(open("o1.sav", "rb"))
model2 = pickle.load(open("o2.sav", "rb"))
model3 = pickle.load(open("o3.sav", "rb"))

def method_call(audio_file,s1,s2):
    header = 'filename rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    header += f' f0 h_f0 h_spec_cent h_rmse h_rolloff h_rms_flat_ratio p_spec_cent p_rmse p_rolloff p_rms_flat_ratio'
    header += f' spec_cent_f0_ratio h_spec_cent_f0_ratio p_spec_cent_f0_ratio'
    header += f' mfcc1_by_mfcc2 mfcc2_by_mfcc3 mfcc3_by_mfcc4'
    
    for i in range(1, 13):
        header += f' chroma_stft{i}'
    for i in range(1, 21):
        header += f' mfcc{i}'
    for i in range(1, 21):
        header += f' mfcc_var{i}'
    for i in range(1, 21):
        header += f' d1_mfcc{i}'
    for i in range(1, 21):
        header += f' d2_mfcc{i}'
    for i in range(1, 21):
        header += f' d3_mfcc{i}'
    for i in range(1, 21):
        header += f' d4_mfcc{i}'
    for i in range(1, 21):
        header += f' d5_mfcc{i}'
    header += ' label'
    header = header.split()


    file = open('test.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    y, sr = librosa.load(audio_file, mono=True, duration=None)
    y_harmonic = librosa.effects.harmonic(y)
    y_percussive = librosa.effects.percussive(y)
    rmse = librosa.feature.rms(y=y)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    f0 = librosa.yin(y, 10, 44100)
    h_f0 = librosa.yin(y_harmonic, 10, 44100 )

    h_spec_cent = librosa.feature.spectral_centroid(y=y_harmonic, sr=sr)
    h_rmse = librosa.feature.rms(y=y_harmonic)
    h_rolloff = librosa.feature.spectral_rolloff(y=y_harmonic, sr=sr)

    p_spec_cent = librosa.feature.spectral_centroid(y=y_percussive, sr=sr)
    p_rmse = librosa.feature.rms(y=y_percussive)
    p_rolloff = librosa.feature.spectral_rolloff(y=y_percussive, sr=sr)

    flatness = librosa.feature.spectral_flatness(y=y)
    h_rms_flat_ratio = h_rmse/flatness
    p_rms_flat_ratio = p_rmse/flatness


    spec_cent_f0_ratio = spec_cent/f0
    h_spec_cent_f0_ratio = h_spec_cent/f0
    p_spec_cent_f0_ratio = p_spec_cent/f0
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    
    mfcc1_by_mfcc2 = mfcc[0]/mfcc[1]
    mfcc2_by_mfcc3 = mfcc[1]/mfcc[2]
    mfcc3_by_mfcc4 = mfcc[2]/mfcc[3]

    
    d1_mfcc = librosa.feature.delta(mfcc)
    d2_mfcc = librosa.feature.delta(mfcc, order = 2)
    d3_mfcc = librosa.feature.delta(mfcc, order = 3)
    d4_mfcc = librosa.feature.delta(mfcc, order = 4)
    d5_mfcc = librosa.feature.delta(mfcc, order = 5)

    to_append = f'{audio_file} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)}'
    to_append += f' {np.mean(rolloff)} {np.mean(zcr)} {np.mean(f0)} {np.mean(h_f0)} {np.mean(h_spec_cent)}'
    to_append += f' {np.mean(h_rmse)} {np.mean(h_rolloff)} {np.mean(h_rms_flat_ratio)} {np.mean(p_spec_cent)} {np.mean(p_rmse)} {np.mean(p_rolloff)}'
    to_append += f' {np.mean(p_rms_flat_ratio)} {np.mean(spec_cent_f0_ratio)}  {np.mean(h_spec_cent_f0_ratio)}'
    to_append += f' {np.mean(p_spec_cent_f0_ratio)} {np.mean(mfcc1_by_mfcc2)} {np.mean(mfcc2_by_mfcc3)} {np.mean(mfcc3_by_mfcc4)}'   

    for e in chroma_stft:
        to_append += f' {np.mean(e)}'
    for e in mfcc:
        to_append += f' {np.mean(e)}'
    for e in mfcc:
        to_append += f' {np.var(e)}'
    for e in d1_mfcc:
        to_append += f' {np.mean(e)}'
    for e in d2_mfcc:
        to_append += f' {np.mean(e)}'
    for e in d3_mfcc:
        to_append += f' {np.mean(e)}'
    for e in d4_mfcc:
        to_append += f' {np.mean(e)}'
    for e in d5_mfcc:
        to_append += f' {np.mean(e)}'
    to_append += f''
    file = open('test.csv', 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(to_append.split())
            
  
    # Read the csv file back
    df = pd.read_csv('test.csv')

     
    df['G'] = s1
    df['A'] = int(s2)
    print(df)
    
    df=df.drop(columns=['filename', 'label'])
    print(df.head(5))
    
    df = df.join(pd.get_dummies(df['G']))
    df=df.drop(columns=['G'])
    
    if s1 =="m":
        print("Inside IF")
        df['m']=1
        df['f']=0
    else:
        print("Inside ELSE")
        df['f']=1
        df['m']=0           
    print(df)
    single1 = model1.predict_proba(df.tail(1))   
    single2 = model2.predict_proba(df.tail(1))  
    single3 = model3.predict_proba(df.tail(1)) 
    single= (single1+single2+single3)/3       
    if single[0][0] > 0.5:
        output='The person is HEALTHY with a confidence score of : ' + str(round(single[0][0]*100,20)+10) + '%'
    else:
        output='The person is UNHEALTHY with a confidence score of : ' + str(round(single[0][1]*100,2)) + '%'     

    return output



app = Flask(__name__)
@app.route("/", methods=['POST', 'GET'])
def method1():
    if request.method == "POST":
        f = request.files['audio_data']
        audio_file = 'audio.wav'
        with open(audio_file, 'wb') as audio:
            f.save(audio)
            f.close()
            print(f)
            s1= request.form['gender']
            s2= int(request.form['age'])
            output=method_call(audio_file,s1,s2)          
            #print(df.head(5))
           # output = df['chroma_stft'].to_list()
            response = app.response_class(
                response=json.dumps(output),
                status=200,
                mimetype='application/json'
            )
            return response
    else:
        return render_template('index.html',output1="Output here")
    
    
@app.route('/upload',methods=['POST'])
def submit_upload():
    if request.method == 'POST':
        return render_template("upload.html")  

@app.route("/submit_upload", methods=['POST', 'GET'])
def method_upload():
    
    if request.method == "POST":
        f = request.files['userfile']
        f.save(f.filename)
        print(f)
        s1=request.form.get('gender')
        s2=request.form['query2']
        print(s1,s2)

        audio_file=f.filename
        output=method_call(audio_file,s1,s2)
        return render_template('upload.html',query2 = request.form['query2'],output=output)
    else:
        return render_template('upload.html')
    
if __name__ == "__main__":
    app.jinja_env.auto_reload = True
    app.run( debug=False)
