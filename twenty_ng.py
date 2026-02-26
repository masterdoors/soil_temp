from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from boosted_forest import CascadeBoostingClassifier
from sklearn import metrics

newsgroups_train = fetch_20newsgroups(subset='train',
                                      remove=('headers', 'footers', 'quotes')
                                      )

vectorizer = TfidfVectorizer()
vectors_train = vectorizer.fit_transform(newsgroups_train.data)

#model = GradientBoostingClassifier(loss = "log_loss", n_estimators=100,  max_depth=1)
model = CascadeBoostingClassifier(loss = "multinomial", n_layers=100, n_estimators = 10, max_depth=1, n_iter_no_change = None, validation_fraction = 0.1, learning_rate = 0.1,hidden_size = 80,verbose=1, n_trees=20,batch_size = 1000)

model.fit(vectors_train,newsgroups_train.target)

newsgroups_test = fetch_20newsgroups(subset='test',
                                      remove=('headers', 'footers', 'quotes')
                                      )

vectors_test = vectorizer.transform(newsgroups_test.data)

pred = model.predict(vectors_test)
pred_train = model.predict(vectors_train)

print(metrics.f1_score(pred_train, newsgroups_train.target, average='macro'))
print(metrics.f1_score(pred, newsgroups_test.target, average='macro'))