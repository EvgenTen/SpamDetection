import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score, accuracy_score

# read data in CSV format
# Change here
data_read = pd.read_csv("D:\Mega\Docs\ira\ira limudim\lemidat mehona\project\SpamDetection\spam.csv",
                        encoding='latin')

data_read.rename(columns={'v1': 'Label', 'v2': 'Text'}, inplace=True)
data_read['numLabel'] = data_read['Label'].map({'ham': 0, 'spam': 1})
data_read['Count'] = 0

for i in np.arange(0, len(data_read.Text)):
    data_read.loc[i, 'Count'] = len(data_read.loc[i, 'Text'])
print("\n")
print("Unique val in the Class set: ", data_read.Label.unique())
print("\n")
ham = data_read[data_read.numLabel == 0]
ham_count = pd.DataFrame(pd.value_counts(ham['Count'], sort=True).sort_index())
print("Number of 'ham' sms-messages in data set:", ham['Label'].count())
spam = data_read[data_read.numLabel == 1]
spam_count = pd.DataFrame(pd.value_counts(spam['Count'], sort=True).sort_index())
print("Number of 'spam' sms-messages in data set:", spam['Label'].count())

# Remove stopwords of English
stopdata = set(stopwords.words("english"))

# Vectorizer
vectorizer = CountVectorizer(stop_words=stopdata, binary=True)
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(data_read.Text)
y = data_read.numLabel

#  test train split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, train_size=0.70, random_state=None)

# Show the result of the split

print("\n")
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))
print("\n")


def train_classifier(clf, X_train, y_train):
    clf.fit(X_train, y_train)

def predict_labels(clf, features):
    return (clf.predict(features))

clf = AdaBoostClassifier(n_estimators=100)

print("Adaboost:\n")
train_classifier(clf, X_train, y_train)
y_pred = predict_labels(clf, X_test)
accuracy_score = accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
f1_score = f1_score(y_test, y_pred)
print("Accuracy in %:")
print(accuracy_score * 100)
print("\n")
print("F1 Score:")
print(f1_score)




