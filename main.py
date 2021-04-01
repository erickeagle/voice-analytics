
import parselmouth
from parselmouth.praat import call
import librosa
import pandas as pd
import numpy as np


import os
from PIL import Image 
import pathlib
import csv 
from flask import Flask, make_response
from flask import json
from flask import request
from flask import render_template

import csv
import pickle


model_m1 = pickle.load(open("rf_m1.sav", "rb"))
model_m2 = pickle.load(open("rf_m2.sav", "rb"))
model_m3 = pickle.load(open("rf_m3.sav", "rb"))
model_m4 = pickle.load(open("rf_m4.sav", "rb"))
model_m5 = pickle.load(open("rf_m5.sav", "rb"))

model_f1 = pickle.load(open("rf_f1.sav", "rb"))
model_f2 = pickle.load(open("rf_f2.sav", "rb"))
model_f3 = pickle.load(open("rf_f3.sav", "rb"))
model_f4 = pickle.load(open("rf_f4.sav", "rb"))
model_f5 = pickle.load(open("rf_f5.sav", "rb"))

# for triming the audio
from pydub import AudioSegment
from audioclipextractor import AudioClipExtractor, SpecsParser
AudioSegment.converter=AudioSegment.ffmpeg = 'ffmpeg.exe'
AudioSegment.ffprobe   = 'ffprobe.exe'
def audioTrim(file,t):
    
    t_trim=(t-1.5)/2

    ext = AudioClipExtractor(file,'ffmpeg.exe')


    specs='''
    {} {} 
    '''.format(t_trim, 1.5+t_trim)

    # Extract the clips according to the specs and save them as a zip archive
    ext.extract_clips(specs)


    sound = AudioSegment.from_mp3("clip1.mp3")
    sound.export("audio_trim.wav", format="wav")
    return ("audio_trim.wav")

# for prediction

sr=50000


def method_call(audioname,s1,s2):
    header = 'filename rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    header += f' f0 h_f0'
    header += f' h_spec_cent h_rmse h_rolloff h_rms_flat_ratio'
    header += f' p_spec_cent p_rmse p_rolloff p_rms_flat_ratio'
    header += f' spec_cent_f0_ratio h_spec_cent_f0_ratio p_spec_cent_f0_ratio'
    header += f' mfcc1_by_mfcc2 mfcc2_by_mfcc3 mfcc3_by_mfcc4'
    header += f' localJitter localabsoluteJitter rapJitter ppq5Jitter'
    header += f' localShimmer localdbShimmer apq3Shimmer aqpq5Shimmer apq11Shimmer'
    header += f' hnr_f0min hnr_125'
    
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


    Cnt=0
    file = open('test.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
        Cnt+=1

        y, sr = librosa.load(audioname, mono=True, duration=None)
        y_harmonic = librosa.effects.harmonic(y)
        y_percussive = librosa.effects.percussive(y)
        rmse = librosa.feature.rms(y=y,  frame_length = 4096, hop_length=256)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=4096, hop_length=256)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=4096, hop_length=256)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=4096, hop_length=256)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=4096,  hop_length=256)
        zcr = librosa.feature.zero_crossing_rate(y, frame_length = 4096, hop_length=256)
        f0 = librosa.yin(y, 70, 50000, sr = sr, frame_length = 4096,   hop_length=256)
        h_f0 = librosa.yin(y_harmonic, 70, 50000, sr = sr, frame_length = 4096)
    
        h_spec_cent = librosa.feature.spectral_centroid(y=y_harmonic, sr=sr, n_fft=4096, hop_length=256)
        h_rmse = librosa.feature.rms(y=y_harmonic, frame_length = 4096, hop_length=256)
        h_rolloff = librosa.feature.spectral_rolloff(y=y_harmonic, sr=sr, n_fft=4096, hop_length=256)
    
        p_spec_cent = librosa.feature.spectral_centroid(y=y_percussive, sr=sr, n_fft=4096, hop_length=256)
        p_rmse = librosa.feature.rms(y=y_percussive, frame_length = 4096, hop_length=256)
        p_rolloff = librosa.feature.spectral_rolloff(y=y_percussive, sr=sr, n_fft=4096, hop_length=256)
    
        flatness = librosa.feature.spectral_flatness(y=y, n_fft=4096, hop_length=256)
        h_rms_flat_ratio = h_rmse/flatness
        p_rms_flat_ratio = p_rmse/flatness
    
    
        spec_cent_f0_ratio = spec_cent/f0
        h_spec_cent_f0_ratio = h_spec_cent/f0
        p_spec_cent_f0_ratio = p_spec_cent/f0
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=4096, hop_length=256, n_mels = 128)
        if mfcc[1][0]!=0:
            mfcc1_by_mfcc2 = mfcc[0]/mfcc[1]
        else:
            mfcc1_by_mfcc2 = mfcc[0]
        if mfcc[2][0]!=0:
            mfcc2_by_mfcc3 = mfcc[1]/mfcc[2]
        else:
            mfcc2_by_mfcc3 = mfcc[1]
        if mfcc[3][0]!=0:
            mfcc3_by_mfcc4 = mfcc[2]/mfcc[3]
        else:
            mfcc3_by_mfcc4 = mfcc[2]
    
        
        d1_mfcc = librosa.feature.delta(mfcc)
        d2_mfcc = librosa.feature.delta(mfcc, order = 2)
        d3_mfcc = librosa.feature.delta(mfcc, order = 3)
        d4_mfcc = librosa.feature.delta(mfcc, order = 4)
        d5_mfcc = librosa.feature.delta(mfcc, order = 5)
    
        sound = parselmouth.Sound(audioname) # read the sound
        f0min = 70 
        f0max = 50000
        unit = "Hertz"
        pitch = call(sound, "To Pitch", 0.0, f0min, f0max)
        pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)#create a praat pitch object
        localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 1/(f0min-5), 1.5)
        localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 1/(f0min-5), 1.5)
        rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 1/(f0min-5), 1.5)
        ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 1/(f0min-5), 1.5)
        localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 1/(f0min-5), 1.3, 1.6)
        localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 1/(f0min-5), 1.3, 1.6)
        apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 1/(f0min-5), 1.3, 1.6)
        aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 1/(f0min-5), 1.3, 1.6)
        apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 1/(f0min-5), 1.3, 1.6)
        harmonicity_f0min = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 4.5)
        hnr_f0min = call(harmonicity_f0min, "Get mean", 0, 0)
        harmonicity125 = call(sound, "To Harmonicity (cc)", 0.01, 125, 0.1, 4.5)
        hnr_125 = call(harmonicity125, "Get mean", 0, 0)
    
        to_append = f'{audioname} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
        to_append += f' {np.mean(f0)} {np.mean(h_f0)}'
        to_append += f' {np.mean(h_spec_cent)} {np.mean(h_rmse)} {np.mean(h_rolloff)} {np.mean(h_rms_flat_ratio)}'
        to_append += f' {np.mean(p_spec_cent)} {np.mean(p_rmse)} {np.mean(p_rolloff)} {np.mean(p_rms_flat_ratio)}'
        to_append += f' {np.mean(spec_cent_f0_ratio)}  {np.mean(h_spec_cent_f0_ratio)} {np.mean(p_spec_cent_f0_ratio)}'
        to_append += f' {np.mean(mfcc1_by_mfcc2)} {np.mean(mfcc2_by_mfcc3)} {np.mean(mfcc3_by_mfcc4)}'   
        to_append += f' {localJitter} {localabsoluteJitter} {rapJitter} {ppq5Jitter}'
        to_append += f' {localShimmer} {localdbShimmer} {apq3Shimmer} {aqpq5Shimmer} {apq11Shimmer}'
        to_append += f' {hnr_f0min} {hnr_125}'
    
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
    df=df.drop(columns=['chroma_stft1', 'chroma_stft2', 'chroma_stft3', 'chroma_stft4', 'chroma_stft5', 'chroma_stft6', 'chroma_stft7', 'chroma_stft8', 'chroma_stft9', 'chroma_stft10', 'chroma_stft11', 'chroma_stft12'])
    df=df.drop(columns = ['d1_mfcc14', 'd1_mfcc15', 'd1_mfcc16', 'd1_mfcc17', 'd1_mfcc18', 'd1_mfcc19', 'd1_mfcc20', 
                          'd2_mfcc14', 'd2_mfcc15', 'd2_mfcc16', 'd2_mfcc17', 'd2_mfcc18', 'd2_mfcc19', 'd2_mfcc20', 
                          'd3_mfcc1', 'd3_mfcc2', 'd3_mfcc3', 'd3_mfcc4', 'd3_mfcc5', 'd3_mfcc6', 'd3_mfcc7', 'd3_mfcc8', 'd3_mfcc9', 'd3_mfcc10', 'd3_mfcc11', 'd3_mfcc12', 'd3_mfcc13', 'd3_mfcc14', 'd3_mfcc15', 'd3_mfcc16', 'd3_mfcc17', 'd3_mfcc18', 'd3_mfcc19', 'd3_mfcc20', 
                          'd4_mfcc1','d4_mfcc2','d4_mfcc3','d4_mfcc4','d4_mfcc5','d4_mfcc6', 'd4_mfcc7', 'd4_mfcc8', 'd4_mfcc9', 'd4_mfcc10', 'd4_mfcc11', 'd4_mfcc12', 'd4_mfcc13', 'd4_mfcc14', 'd4_mfcc15', 'd4_mfcc16', 'd4_mfcc17', 'd4_mfcc18', 'd4_mfcc19', 'd4_mfcc20', 
                          'd5_mfcc1', 'd5_mfcc2', 'd5_mfcc3', 'd5_mfcc4', 'd5_mfcc5','d5_mfcc6', 'd5_mfcc7', 'd5_mfcc8', 'd5_mfcc9', 'd5_mfcc10', 'd5_mfcc11', 'd5_mfcc12', 'd5_mfcc13', 'd5_mfcc14', 'd5_mfcc15', 'd5_mfcc16', 'd5_mfcc17', 'd5_mfcc18', 'd5_mfcc19', 'd5_mfcc20'])

     
    df['G'] = s1
    df['A'] = int(s2)
    df=df.drop(columns=['filename', 'label'])
    df = df.join(pd.get_dummies(df['G']))
    df=df.drop(columns=['G'])
    
    if s1 =="m":
        print("Inside IF")
        df['m']=1
        df['f']=0
        single1 = model_m1.predict_proba(df.tail(1))   
        single2 = model_m2.predict_proba(df.tail(1))  
        single3 = model_m3.predict_proba(df.tail(1)) 
        single4 = model_m4.predict_proba(df.tail(1)) 
        single5 = model_m5.predict_proba(df.tail(1))
    else:
        print("Inside ELSE")
        df['f']=1
        df['m']=0 
        single1 = model_f1.predict_proba(df.tail(1))   
        single2 = model_f2.predict_proba(df.tail(1))  
        single3 = model_f3.predict_proba(df.tail(1)) 
        single4 = model_f4.predict_proba(df.tail(1)) 
        single5 = model_f5.predict_proba(df.tail(1))          

    
     
    single= (single1+single2+single3+single4+single5)/5      
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
            data,sr=librosa.load(audio_file,sr=50000)
            t=librosa.get_duration(data,sr)
            '''
            if t>1.5:
                audio_file=audioTrim(audio_file,t)
                print(audio_file)
                output=method_call(audio_file,s1,s2)  
            else:
                output="Please record Again length should be more than equal to 1.5 sec" 
    
         
            '''
            
                    

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
        '''
        data,sr=librosa.load(audio_file,sr=50000)
        t=librosa.get_duration(data,sr)
        if t>1.5:
            audio_file=audioTrim(audio_file,t)
            print(audio_file)
            output=method_call(audio_file,s1,s2)  
        else:
            output="Please record Again length should be more than equal to 1.5 sec" 
    '''
        return render_template('upload.html',query2 = request.form['query2'],output=output)
    else:
        return render_template('upload.html')
    
if __name__ == "__main__":
    app.jinja_env.auto_reload = True
    app.run( debug=False)
