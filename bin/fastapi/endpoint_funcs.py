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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import chi2
from pymongo import MongoClient, database
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
import joblib

# define the class encodings and reverse encodings
classes = {'general': 0}  # modify this later

# function to train and load the model during startup


def load_model():
    global clf
    # load the model and save it in clf


# function to predict the category using the model
def predict(query_data):
    x = list(query_data.dict().values())
    prediction = clf.predict([x])[0]
    print(f"Model prediction: {classes[prediction]}")
    return classes[prediction]

# function to retrain the model as part of the feedback loop


def retrain():
    live_data = get_mongo_data()
    live_data.to_csv("live_data.csv", index=False)
    X, y = prepare_data(live_data)
    loaded_model = joblib.load("model.sav")
    loaded_model.partial_fit(X, y)
    print("trained successfully")

    # save the model locally
    # model.save()


def prepare_data(df):
    df.fillna(0, inplace=True)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    df['category_id'] = df['category'].factorize()[0]
    category_id_df = df[['category', 'category_id']
                        ].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'category']].values)

    df.groupby('category').category_id.count()
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                            stop_words='english')

    features = tfidf.fit_transform(df.description.values.astype('U')).toarray()

    labels = df.category_id
    return features, labels


def train(clf=SGDClassifier(random_state=42)):

    df = pd.read_csv("../combined.csv")
    df = df[['source', 'description', 'title', 'category']]
    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.33,
                                                        random_state=42)
    clf.partial_fit(X_train, y_train, classes=np.unique(y))
    y_pred = clf.predict(X_test)
    score = f1_score(y_test, y_pred, average='macro')
    print("Trained with an accuracy:{}".format(score))
    joblib.dump(clf, "model.sav")


def load_model():
    loaded_model = joblib.load("./fastapi/models/model.sav")


# train()
# retrain()


def get_mongo_data():
    try:
        conn = MongoClient('mongodb://root:example@localhost:27017', 27017)
        # print("connected sucessfully")
    except:
        print("could not connect to mongodb")
        return

    db = conn.database
    users = db.users
    cursor = users.find()
    live_data = pd.DataFrame(list(cursor))
    live_data = live_data.iloc[:300, :]
    live_data.drop_duplicates(subset=["_id"], inplace=True)
    return live_data.drop(["_id"], axis=1)


# train()
# retrain()


def test():
    train = pd.read_csv('../combined.csv')
    new_data = pd.read_csv("live_data.csv")

    X1, y1 = prepare_data(train)
    X2, y2 = prepare_data(new_data)

    print(X1.shape)
    print(X2.shape)


train()
retrain()
