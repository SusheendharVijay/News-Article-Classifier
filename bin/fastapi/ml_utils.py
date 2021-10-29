from pymongo import MongoClient, database
import re
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import joblib  # saving the model
# function to split the data for cross-validation
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
# function for transforming documents into counts
from sklearn.feature_extraction.text import CountVectorizer
# function for encoding categories
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier


def normalize_text(s):
    s = str(s)
    s = s.lower()

    # remove punctuation that is not word-internal (e.g., hyphens, apostrophes)
    s = re.sub('\s\W', ' ', s)
    s = re.sub('\W\s', ' ', s)

    # make sure we didn't introduce any double spaces
    s = re.sub('\s+', ' ', s)

    return s


def process_data(news):
    for col in ['title', 'description']:
        news[col] = [normalize_text(s) for s in news[col]]

    # pull the data into vectors
    vec1 = CountVectorizer()
    vec2 = CountVectorizer()
    x1 = vec1.fit_transform(news['title'])
    x2 = vec2.fit_transform(news['description'])

    X = np.concatenate((x1.toarray(), x2.toarray()), axis=1)
    encoder = LabelEncoder()
    y = encoder.fit_transform(news['category'])
    joblib.dump(encoder, "encoder.sav")
    return X, y, vec1, vec2


def process_new_data(news, vec1, vec2):
    for col in ['title', 'description']:
        news[col] = [normalize_text(s) for s in news[col]]

    # pull the data into vectors
    x1 = vec1.transform(news['title'])
    x2 = vec2.transform(news['description'])

    X = np.concatenate((x1.toarray(), x2.toarray()), axis=1)
    encoder = joblib.load('encoder.sav')
    y = encoder.fit_transform(news['category'])
    return X, y


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


def retrain():
    live_data = get_mongo_data()
    vec1 = joblib.load("vec1.sav")
    vec2 = joblib.load("vec2.sav")
    live_data.to_csv("live_data.csv", index=False)
    X, y = process_new_data(live_data, vec1, vec2)
    loaded_model = joblib.load("model.sav")

    loaded_model.partial_fit(X, y)
    print("retrained successfully")

    combined = pd.read_csv("../combined.csv")
    X_train, y_test = process_new_data(combined, vec1, vec2)

    test_preds = loaded_model.predict(X_train)
    score = f1_score(y_test, test_preds, average='macro')
    print("Accuracy after retraining:", score)


# temp

#
def train():
    clf = SGDClassifier()
    combined = pd.read_csv("../combined.csv")
    X, y, vec1, vec2 = process_data(combined)

    # split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf.partial_fit(x_train, y_train, classes=np.unique(y))
    preds = clf.predict(x_test)
    score = f1_score(y_test, preds, average='macro')
    print("Accuracy:", score)

    saved_model = SGDClassifier()
    saved_model.partial_fit(X, y, classes=np.unique(y))
    joblib.dump(saved_model, "model.sav")
    print("Saved model!")
    joblib.dump(vec1, "vec1.sav")
    joblib.dump(vec2, "vec2.sav")
    print("Saved transformers!")


def predict(query_data):
    print(query_data)
    news = pd.DataFrame(
        [[query_data.title, query_data.description]], columns=['title', 'description'])

    vec1 = joblib.load("vec1.sav")
    vec2 = joblib.load("vec2.sav")
    loaded_model = joblib.load("model.sav")

    print(news.head())
    for col in ['title', 'description']:
        news[col] = [normalize_text(s) for s in news[col]]

    # pull the data into vectors
    x1 = vec1.transform(news['title'])
    x2 = vec2.transform(news['description'])

    X = np.concatenate((x1.toarray(), x2.toarray()), axis=1)
    prediction = loaded_model.predict(X)
    encoder = joblib.load('encoder.sav')
    category = encoder.inverse_transform(prediction)
    print("Prediction", category[0])
    return category[0]


# train()
if __name__ == '__main__':
    train()
    retrain()
