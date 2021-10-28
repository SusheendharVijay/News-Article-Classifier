from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import os
from sklearn.feature_selection import chi2
from pymongo import MongoClient, database
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()
# define the class encodings and reverse encodings
classes = {'general': 0}  # modify this later

# function to train and load the model during startup


def load_model():
    global clf
    # load the model and save it in clf


# function to predict the flower using the model
def predict(query_data):
    x = list(query_data.dict().values())
    prediction = clf.predict([x])[0]
    print(f"Model prediction: {classes[prediction]}")
    return classes[prediction]

# function to retrain the model as part of the feedback loop


def retrain():
    # get data from api
    # clean it
    try:
        conn = MongoClient('mongodb://root:example@localhost:27017', 27017)
        # print("connected sucessfully")
    except:
        print("could not connect to mongodb")
        return

    db = conn.database
    users = db.users
    cursor = users.find()
    mongo_docs = list(cursor)
    # print(mongo_docs)
    docs = pd.DataFrame(columns=[])
    for num, doc in enumerate(mongo_docs):
        doc["_id"] = str(doc["_id"])
        doc_id = doc["_id"]
        series_obj = pd.Series(doc, name=doc_id)
        docs = docs.append(series_obj)
    docs.drop_duplicates(subset="_id")
    docs.to_csv("retrain_data.csv", ",")
    X = docs.drop(["_id", "category"])
    y = docs.category.values
    # tfidf + label encoding
    # retrain the model with partial fit
    # X_train, X_test, y_train, y_test = train_test_split(, labels, df.index,
    #                                                     test_size=0.33,
    #                                                     random_state=42)
    clf.fit(docs)
    # model.partial_fit(docs)
    # save the model locally
    # model.save()


retrain()
