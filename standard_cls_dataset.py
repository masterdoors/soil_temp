
import argparse
import os
from os.path import join
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn import preprocessing
from sklearn import datasets
from sklearn.impute import SimpleImputer   
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer

import math
from scipy import sparse
import pandas as pd
from scipy.sparse import csr_matrix
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords

import re
from sklearn.datasets import fetch_20newsgroups

import xgboost as xgb
from boosted_forest import CascadeBoostingRegressor
from deepforest import CascadeForestRegressor
from catboost import CatBoostRegressor
from residual import ResNetCls
from  sklearn.ensemble import RandomForestClassifier

def load20ng():
    newsgroups_train = fetch_20newsgroups(subset='train',
                                        remove=('headers', 'footers', 'quotes')
                                        )
    
    newsgroups_test = fetch_20newsgroups(subset='test',
                                        remove=('headers', 'footers', 'quotes')
                                        )
    vectorizer = TfidfVectorizer()
    vectors_train = vectorizer.fit_transform(newsgroups_train.data)    
    vectors_test = vectorizer.transform(newsgroups_test.data)    
    return vectors_train, newsgroups_train.target,vectors_test, newsgroups_test.target

def load_cifar_batch(file):
    with open(file, 'rb') as f:
        data_dict = pickle.load(f, encoding='latin1')
        return data_dict['data'], np.array(data_dict['fine_labels'])

def loadCifar10():
    batches = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']

    bs = [load_cifar_batch(os.path.join('cifar-10-batches-py', b)) for b in batches]
    X_train = np.vstack([b[0] for b in bs])
    y_train = np.hstack([b[1] for b in bs])
    X_test, y_test = load_cifar_batch(os.path.join('cifar-10-batches-py', 'test_batch'))

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    return X_train, y_train, X_test,y_test

def loadCifar100():
    X_train, y_train = load_cifar_batch(os.path.join('cifar-100-python', 'train'))

    X_test,y_test  = load_cifar_batch(os.path.join('cifar-100-python', 'test'))

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    return X_train, y_train, X_test,y_test

def loadBNP():
    tbl = pd.read_csv("BNP/train.csv.tar.gz",sep=',',compression = 'infer')
    mtx = tbl.to_numpy()
    x_mtx = mtx[:,2:]
    y_mtx = mtx[:,1]
    y = np.asarray(y_mtx,dtype=int) + 1

    arrs_to_conc = []

    def my_func():
        for i in range(x_mtx.shape[1]):
            try:
                arr = np.unique(x_mtx[:,i].astype(float))
            except:
                arr = np.unique(x_mtx[:,i].astype(str))    

            if len(arr) < 40000:
                digitized_arr = LabelEncoder().fit_transform(x_mtx[:,i])
                if isinstance(arr[0],float) and math.isnan(arr[0]):
                    nan_idx = digitized_arr == 0
                    digitized_arr[nan_idx] = len(arr) * 2            
                coded_arr =  sparse.lil_matrix(OneHotEncoder(sparse_output=True,handle_unknown='ignore').fit_transform(digitized_arr.reshape(-1, 1)))

                arrs_to_conc.append(coded_arr)
            else:
                arrs_to_conc.append(sparse.lil_matrix(x_mtx[:,i].reshape(-1, 1),dtype=float))

        return sparse.hstack(arrs_to_conc)    

    res_arr = my_func()

    res_arr = SimpleImputer(strategy='most_frequent',copy=False).fit_transform(res_arr)
    x = normalize(res_arr,axis=0)
    x_train, x_test, y_train, y_test = train_test_split(x,y)
    return x_train,y_train,x_test,y_test 

def loadIMDB():
    nltk.download('omw-1.4')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt_tab')

    df = pd.read_csv('IMDB Dataset.csv')

    df['review'] = df['review'].astype(str)
    reviews = df['review']

    def remove_tags(string):
        result = re.sub('<.*?>','',string)
        return result

    reviews = reviews.apply(lambda i: remove_tags(i))

    reviews = reviews.str.replace(r'[^\w\d\s]', ' ',regex=True)

    reviews = reviews.str.replace(r'\s+', ' ',regex=True)

    reviews = reviews.str.replace(r'^\s+|\s+?$', '',regex=True)

    stopwords = nltk.corpus.stopwords.words("english")
    reviews =reviews.apply(lambda x: ' '.join(word for word in x.split() if word not in stopwords))

    reviews = reviews.str.lower()

    wn_lemmatizer = WordNetLemmatizer()
    reviews = reviews.apply(lambda x: ' '.join(wn_lemmatizer.lemmatize(word) for word in x.split()))

    tokenize_words = []

    for review in reviews:
        words = word_tokenize(review)
        for word in words:
            tokenize_words.append(word)


    count_vec = CountVectorizer(binary=False,ngram_range=(2,3),max_features=20000)

    x = count_vec.fit_transform(reviews).toarray()

    count_vec_x = x.astype(np.int8)

    tfid_vec = TfidfVectorizer(binary=False,ngram_range=(2,3),max_features=20000)

    tfid_x = tfid_vec.fit_transform(reviews).toarray()

    sentiment = pd.get_dummies(df,columns=['sentiment'])

    sentiment.drop(['review','sentiment_negative'],axis = 1,inplace=True)

    sentiment.rename(columns = {'sentiment_positive':'sentiment'}, inplace = True)

    sentiment = np.squeeze(sentiment).to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(count_vec_x, sentiment, test_size = 0.20)
    return x_train, y_train, x_test, y_test


def loadMNIST():
    digits = datasets.load_digits()

    n_samples = len(digits.images)

    data = digits.images.reshape((n_samples, -1))

    Y =  np.asarray(digits.target).astype('int64')


    x = preprocessing.normalize(data, copy=False, axis = 0)

    x_train, x_validate, Y_train, Y_validate = train_test_split(
        x, Y, test_size=0.5, shuffle=True
    )
    return x_train, Y_train, x_validate, Y_validate

def printf(*args, fname="log.txt"):
    with open(join("test_outputs",fname),"a+") as f:
        for a in args:
            f.write(str(a) + " ")
        f.write("\n") 
    print(args) 

def make_modelXGB(max_depth,layers,n_trees,n_estimators):
    return xgb.XGBRegressor(max_depth = max_depth, n_estimators = layers)

def make_modelCAT(max_depth,layers,n_trees,n_estimators):
    return CatBoostRegressor(learning_rate = 0.1,num_trees = layers,max_depth = max_depth)

def make_modelCascade(max_depth,layers,n_trees,n_estimators):
    return CascadeForestRegressor(max_depth = max_depth, max_layers = layers, n_estimators=n_estimators,backend="sklearn",criterion='squared_error',n_trees=n_trees,n_tolerant_rounds = 100)

def make_modelBoosted(max_depth,layers,hs,n_trees,n_estimators):
    return CascadeBoostingRegressor(n_layers=layers, n_estimators = n_estimators, max_depth=max_depth, n_iter_no_change = None, validation_fraction = 0.1, learning_rate = 0.1,hidden_size = hs,verbose=1, n_trees=n_trees,batch_size = 1000)

def make_modelRF(max_depth, layers, max_features):
    return RandomForestClassifier(n_estimators=max_depth, max_depth=max_depth, max_features=max_features)

def make_modelResNet(layers):
    return ResNetCls(0.001,layers)    

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="Model (XGB, BOOSTED, CASCADE,CAT,RF,RES)")
parser.add_argument("--dataset", type=str, help="Dataset (20NG, CIFAR10, CIFAR100, BNP, IMDB, MNIST)")

parser.add_argument("--layers", type=int)
parser.add_argument("--max_depth", type=int)
parser.add_argument("--hs", type=int)
parser.add_argument("--n_trees", type=int)
parser.add_argument("--max_features", type=float)
parser.add_argument("--train_speed", type=float)

args = parser.parse_args()
model_ = args.model
dataset = args.dataset

if model_ == "XGB":
    model_name = "XGB"
    model = make_modelXGB()
elif model_ == "CAT Boost":    
    model_name = "CAT"
    model = make_modelCAT()
elif model_ == "BOOSTED":
    model_name = "Boosted Forest"    
    model = make_modelBoosted()
elif model_ == "RF":    
    model_name = "Random Forest"        
    model = make_modelRF()
elif model_ == "RES":    
    model_name = "Residual Network"        
    model = make_modelResNet()
else:
    model_name = "Cascade Forest"    
    model = make_modelCascade() 


if dataset == "20NG":
    x_train, Y_train, x_validate, Y_validate = load20ng()
elif dataset == "CIFAR10":
    x_train, Y_train, x_validate, Y_validate = loadCifar10()    
elif dataset == "CIFAR100":    
    x_train, Y_train, x_validate, Y_validate = loadCifar100()    
elif dataset == "BNP":        
    x_train, Y_train, x_validate, Y_validate = loadBNP()    
elif dataset == "IMDB":            
    x_train, Y_train, x_validate, Y_validate = loadIMDB()    
elif dataset == "MNIST":                
    x_train, Y_train, x_validate, Y_validate = loadMNIST()