#Ravikumar Murthy, Sindoora

import pandas as pd
import math
import numpy as np
import pyodbc
import os
import time
from math import radians, cos, sin, asin, sqrt
import sys
from flask import Flask, flash,render_template, request, redirect, url_for, jsonify
import csv
import base64
from azure.storage.blob import BlobServiceClient, BlobClient, ContentSettings, ContainerClient
import copy
import random
import re
import timeit

app = Flask(__name__)


sub_folders=[]
path=[]      
global_dictionary = {}                                  
dictionary = {}
file_name ={}
global domain
domain='NULL'
path = os.getcwd()+"/20_newsgroups/"

# list and space the stop words from the text file
def stop_words_calc():
    pathcheck('stopwords.txt','Stop Words file') 
    f=open("stopwords.txt", "r") 
    next_line=['\n'] 
    stop_words=f.read()       
    for i in next_line:
        stop_words= stop_words.replace(i,' ')
        li = list(stop_words.split(" "))
    text = ([ f' {x} ' for x in li ]) 
    text.pop() 
    return text 

# load the test files sequentially
def load_test_files(sub_folders):               
    global domain                                 
    while (len(sub_folders)):
        newsgroup = sub_folders[len(sub_folders)-1]
        if len(file_name[newsgroup])== 0:
            sub_folders.pop() 
        else:
            next_file = file_name[newsgroup][len(file_name[newsgroup])-1]
            file_name[newsgroup].pop()
            domain = newsgroup
            text = open_file(path + newsgroup + '/'+ next_file) 
            return text
    domain = 'NULL'
    return domain

# clean the data
def preprocessing(text,stop_words):                                                                
    symbol_remove = ['<','>','?','.','"',')','(','|','-','#','*','+','\'','&','^','`','~','\t','$','%',"'",'!','/','\\','=',',',':']
    text = text.lower()
    email = re.findall(r'[\w\.-]+@[\w\.-]+', text)              ## regex email addresses and removing them
    text = text.replace('\n', ' ')
    for word in email:
        text = text.replace(word,' ')
    for word in stop_words:                                                                    #   Comment this line and the line below to 
        text = text.replace(word,' ')                                                          #   not remove stop words from the dataset
    for word in symbol_remove:
        text = text.replace(word,'')
    text = text.split(' ') 
    if '' in text: 
        text.remove('') 
    if ' ' in text: 
        text.remove(' ') 
    return text

#calculate the posterior probability    
def calculate(_file_, dictionary,sub_folder_list):                                   
    probability = 0.0
    predictor_prob = sum(dictionary.values())
    prior_probability=1/len(sub_folder_list)
    for word in _file_:
        likelihood = dictionary.get(word, 0.001)                                            ### change the laplacian smoothening precision here to increase or decrease accuracy by 2%( range 0.1 to 0.000001)
        probability+= math.log((float(likelihood)*prior_probability)/float(predictor_prob))
    return probability


def open_file(file_location):
    with open(file_location, encoding="latin-1") as datafile:
        text = datafile.read() 
    return text

# train the model
def train_model(sub_folder_list,stop_words):               
    print (".....Training the model.......\n")
    i=1 
    record=[]
    for sub_folder in sub_folder_list:
        print(i/len(sub_folder_list)*100,"% trained") 
        i+=1 
        local_dictionary = {} 
        sub_folder_path = path + sub_folder 
        files = os.listdir(sub_folder_path) 
        training_set =int(len(files)/2)+1                                                   # half the dataset constituents the training files.
        flag = 0 
        for _file_ in files:
            flag+= 1 
            if flag > training_set:
                break 
            opened_file=open_file(sub_folder_path + '/'+_file_)                             # The dataset needed to be opened in Latin-1 Encoding.
            words=preprocessing(opened_file,stop_words) 
            for field in words:
                value = local_dictionary.get(field, 0) 
                value_t = global_dictionary.get(field, 0) 
                local_dictionary[field]=value+1                                             
                global_dictionary[field]=value_t+1 
            record.append(_file_) 
        files=list(set(files).difference(record))                                           # comment this line to test the entire dataset.
        dictionary[sub_folder] = local_dictionary 
        file_name[sub_folder] = files 
    

# test the model
def test_model(sub_folder_list,stop_words):
    print ("\n.......Testing the model........\n")
    loop = 0
    text = 1
    prediction = 0
    sub_folders = copy.deepcopy(sub_folder_list)
    while (text):
        if(loop>0 and loop%493==0):
            print(int(loop/9966*100),'% tested') 
        text = load_test_files(sub_folders)                                                    # loading the test files
        if text =='NULL':
            break 
        loop+= 1 
        confusion_matrix = [] 
        words=preprocessing(text,stop_words) 
        for sub_folder in sub_folder_list:
            confusion_matrix.append(calculate(words,dictionary[sub_folder],sub_folder_list)) 
        predicted=max(confusion_matrix) 
        if domain == sub_folder_list[confusion_matrix.index(predicted)]:
            prediction +=1    
    return prediction,loop-1

#check if the supporting files are present
def pathcheck(path,supporting_file_name):
    if os.path.exists(path):
        print('\n',supporting_file_name,"detected in the current Directory\n") 
    else:
        print('\nDataset not present in parent directory. Load dataset and try again.\n Program exiting!!!') 
        exit() 
 

@app.route('/')
def index():
    return render_template('homepage.html')

@app.route('/myinfo')
def myInfo():
    return render_template('firstpage.html')


@app.route('/actions', methods=["POST","GET"])
def actions():
  if request.method== "POST":
    return render_template('functions.html')


@app.route('/index', methods=["POST","GET"])
def query():
  if request.method== "POST":
    return render_template('index.html',input=1)

@app.route('/test', methods=["POST","GET"])
def query1():
  if request.method== "POST":
    stop_words= stop_words_calc()
    start = timeit.default_timer() 
    pathcheck(path,'20 News Group Dataset') 
    sub_folder_list = os.listdir(path)                   ## list of all sub directories
    domain='NULL' 
    train_model(sub_folder_list,stop_words) 
    prediction,loop= test_model(sub_folder_list,stop_words) 
    accuracy=(prediction/loop*100)
    stop = timeit.default_timer() 
    
    return render_template('index.html',input=2,accuracy=accuracy, timelapsed=int(stop - start))

if __name__ == '__main__':
  app.run(debug=True)
