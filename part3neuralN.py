import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

df = pd.read_csv('updatedBased2.csv')[350:]

X = df['message']
y = df['is_based']

#--------- FIRST WAY-----
#Filtering and tokenizing of stopwords(useless words)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X)

#Make long and short documents share same info and weigh down common words
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

#Train the model using MultinomialNB and the updated X_train
clf = MLPClassifier(solver='lbfgs',hidden_layer_sizes=[10,10],random_state=0).fit(X_train_tf, y)

#-------SECOND WAY----
clf = Pipeline([
                ('count_vect',CountVectorizer()),
                ('tfidf',TfidfTransformer(use_idf=False)),
                ('clf',MLPClassifier(solver='lbfgs',hidden_layer_sizes=[11,11],random_state=0,max_iter=10000))
])
clf.fit(X, y)

def changeString(theStr):
    docs = [theStr]
    X_new_counts = count_vect.transform(docs)
    return tf_transformer.transform(X_new_counts)

print(clf.predict(changeString("I love this subreddit!")))
