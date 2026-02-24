import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from boosted_forest import CascadeBoostingClassifier
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix
import nltk
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

df = pd.read_csv('IMDB Dataset.csv')

df['review'] = df['review'].astype(str)
reviews = df['review']

import re
import string

def remove_tags(string):
    result = re.sub('<.*?>','',string)
    return result

reviews = reviews.apply(lambda i: remove_tags(i))

reviews = reviews.str.replace(r'[^\w\d\s]', ' ',regex=True)

reviews = reviews.str.replace(r'\s+', ' ',regex=True)

reviews = reviews.str.replace(r'^\s+|\s+?$', '',regex=True)

from nltk.corpus import stopwords
stopwords = nltk.corpus.stopwords.words("english")
reviews =reviews.apply(lambda x: ' '.join(word for word in x.split() if word not in stopwords))

reviews = reviews.str.lower()

from nltk.stem import WordNetLemmatizer

wn_lemmatizer = WordNetLemmatizer()
reviews = reviews.apply(lambda x: ' '.join(wn_lemmatizer.lemmatize(word) for word in x.split()))

from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.probability import FreqDist

tokenize_words = []

for review in reviews:
    words = word_tokenize(review)
    for word in words:
        tokenize_words.append(word)

from sklearn.feature_extraction.text import CountVectorizer

count_vec = CountVectorizer(binary=False,ngram_range=(2,3),max_features=20000)

x = count_vec.fit_transform(reviews).toarray()

count_vec_x = x.astype(np.int8)

from sklearn.feature_extraction.text import TfidfVectorizer

tfid_vec = TfidfVectorizer(binary=False,ngram_range=(2,3),max_features=20000)

tfid_x = tfid_vec.fit_transform(reviews).toarray()

sentiment = pd.get_dummies(df,columns=['sentiment'])

sentiment.drop(['review','sentiment_negative'],axis = 1,inplace=True)

sentiment.rename(columns = {'sentiment_positive':'sentiment'}, inplace = True)

sentiment = np.squeeze(sentiment).to_numpy()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(count_vec_x, sentiment, test_size = 0.20)

model = CascadeBoostingClassifier(loss = "binomial", n_layers=100, n_estimators = 10, max_depth=2, n_iter_no_change = None, validation_fraction = 0.1, learning_rate = 0.1,hidden_size = 10,verbose=1, n_trees=10,batch_size = 1000)
#model = GradientBoostingClassifier(loss = "log_loss", n_estimators=100,  max_depth=1)

model.fit(
    csr_matrix(x_train,dtype=float),
    y_train,
)        

y_pred = model.predict(csr_matrix(x_test,dtype=float)) 
y_pred2 = model.predict(csr_matrix(x_train,dtype=float))
mse_score = accuracy_score(y_test.flatten(),y_pred.flatten())
print(accuracy_score(y_test.flatten(),y_pred.flatten()))
print("Outer train: ", accuracy_score(y_train.flatten(),y_pred2.flatten()))


