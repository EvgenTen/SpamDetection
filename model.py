# 1.TFIDF vectors with multinomial naive bayes (MultinomialNB)
# 2. Testing specific messages. We get a new message and use our model to determine which class it belongs to.

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, accuracy_score
from nltk.stem import SnowballStemmer
import matplotlib.pyplot as plt
import string

# read data in CSV format
# Change here
data_read = pd.read_csv("D:\Mega\Docs\ira\ira limudim\lemidat mehona\project\SpamDetection\spam.csv",
                        encoding='latin')

data_read.rename(columns={'v1': 'Label', 'v2': 'Text'}, inplace=True)
data_read['numLabel'] = data_read['Label'].map({'ham': 0, 'spam': 1})
data_read['Count'] = 0


data_read["Label"].value_counts().plot(kind = 'pie', explode = [0, 0.1], figsize = (6, 6), autopct = '%1.2f%%')
plt.ylabel('')
plt.legend(["Ham", "Spam"])
plt.show()

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
tfidf_transformer = TfidfTransformer().fit(x)
msgs_tfidf = tfidf_transformer.transform(x)
y = data_read.numLabel

#print( '{}: {}'.format('you', tfidf_transformer.idf_[vectorizer.vocabulary_['you']]))
#print ('{}: {}'.format('hey', tfidf_transformer.idf_[vectorizer.vocabulary_['hey']]))

X_train, X_test, y_train, y_test = train_test_split(msgs_tfidf, y, test_size=0.30, train_size=0.70, random_state=None)

# Show the result of the split

print("\n")
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))
print("\n")

def train_classifier(clf, X_train, y_train):
    clf.fit(X_train, y_train)

def predict_labels(clf, features):
    return (clf.predict(features))

model = MultinomialNB(alpha=0.25,fit_prior=True)

print("Multi-NB:\n")
train_classifier(model, X_train, y_train)
y_pred = predict_labels(model, X_test)
accuracy_score = accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
f1_score = f1_score(y_test, y_pred)
print("Accuracy in %:")
print(accuracy_score * 100)
print("\n")
print("F1 Score:")
print(f1_score)

print("\n")
print("Testing specific messages:")
print("\n")
SMS1 = 'URGENT! Your Mobile No 097669047 was awarded a vacation'

print("SMS1 = 'URGENT! Your Mobile No 097679047 was awarded a vacation'\n")

SMS2 = 'Hello my friend, how are you?'

print("SMS2 = 'Hello my friend'\n")



stemmer = SnowballStemmer("english")
def cleantext(data):
    data = data.translate(str.maketrans('', '', string.punctuation))
    words = [stemmer.stem(word) for word in data.split() if word.lower() not in stopwords.words("english")]
    return " ".join(words)

SMS1_clean = cleantext(SMS1)
SMS1_features = vectorizer.transform([SMS1_clean])
prediction1 = model.predict(SMS1_features)

SMS2_clean = cleantext(SMS2)
SMS2_features = vectorizer.transform([SMS2_clean])
prediction2 = model.predict(SMS2_features)

# input SMS3
print('please write a new sentence using words from the top spam words or regular words:')
SMS3 = str(input())
SMS3_clean = cleantext(SMS3)
SMS3_features = vectorizer.transform([SMS3_clean])
prediction3 = model.predict(SMS3_features)

class1 = 'spam' if prediction1 == 1 else 'ham'
class2 = 'spam' if prediction2 == 1 else 'ham'
class3 = 'spam' if prediction3 == 1 else 'ham'

print(f'SMS1 is {class1} .. SMS2 is {class2} .. new sentence is {class3}')

