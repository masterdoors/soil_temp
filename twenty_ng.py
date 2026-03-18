from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from boosted_forest import CascadeBoostingClassifier
import xgboost as xgb
from sklearn import metrics

newsgroups_train = fetch_20newsgroups(subset='train',
                                      remove=('headers', 'footers', 'quotes')
                                      )

vectorizer = TfidfVectorizer()
vectors_train = vectorizer.fit_transform(newsgroups_train.data)

#model = GradientBoostingClassifier(loss = "log_loss", n_estimators=500,  max_depth=2, verbose = 1, max_features = 1.0)
#model = xgb.XGBClassifier(max_depth = 2, n_estimators = 500)
model = CascadeBoostingClassifier(loss = "multinomial", n_layers=100, n_estimators = 10, max_depth=2, n_iter_no_change = None, validation_fraction = 0.1, learning_rate = 0.1,hidden_size = 6,verbose=1, n_trees=6,batch_size = 3000,max_features=0.2)

model.fit(vectors_train,newsgroups_train.target)

newsgroups_test = fetch_20newsgroups(subset='test',
                                      remove=('headers', 'footers', 'quotes')
                                      )

vectors_test = vectorizer.transform(newsgroups_test.data)

pred = model.predict(vectors_test)
pred_train = model.predict(vectors_train)

print(metrics.f1_score(pred_train, newsgroups_train.target, average='macro'))
print(metrics.f1_score(pred, newsgroups_test.target, average='macro'))

#Boosted cascade
#0.7695786722266641
#0.5899076433283406


#H = 8, T = 6
#0.7913410022428231
#0.5935455549731399

#H = 6, T = 7, FS = 0.3

#0.7848727872770672
#0.6052161544107267

#GB
#0.7159331246771841
#0.5962612800821956

#max features: 0.5
#0.7158983557290532
#0.5921981619534766

#max features:
#0.7175644334327981
#0.5969545807487823

#500 layers
#F=0.5
#0.9333861035372996
#0.6113187639443505

#500 layers
#F
#0.9333861035372996
#0.6113187639443505

#500 layers, F 1, tree depth = 2
#0.9751353060295893
#0.6120884127720227

#XGB
#500, td = 1
#0.8884677540496341
#0.6093621187617584
#500 td = 2
#0.9663236905589109
#0.5952373986199563


