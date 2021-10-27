import matplotlib.pyplot as plt
import sklearn
import pickle
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


def convert_data_to_csv():
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
    docs = pd.DataFrame(columns=[])
    for num, doc in enumerate(mongo_docs):
        doc["_id"] = str(doc["_id"])
        doc_id = doc["_id"]
        series_obj = pd.Series(doc, name=doc_id)
        docs = docs.append(series_obj)
    docs.drop_duplicates(subset="_id")
    docs.to_csv("news_data.csv", ",")
    csv_export = docs.to_csv(sep=",")
    return docs

def trainModel():
    df = pd.read_csv("combined.csv")
    df.fillna(0)
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
    model = LogisticRegression(random_state=0)
    labels = df.category_id
    N = 3
    for Category, category_id in sorted(category_to_id.items()):
        features_chi2 = chi2(features, labels == category_id)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        print("# '{}':".format(Category))
        print("  . Most correlated unigrams:\n       . {}".format(
            '\n       . '.join(unigrams[-N:])))
        print("  . Most correlated bigrams:\n       . {}".format(
            '\n       . '.join(bigrams[-N:])))
        SAMPLE_SIZE = int(len(features) * 0.3)
        np.random.seed(0)
        indices = np.random.choice(range(len(features)), size=SAMPLE_SIZE,
                                   replace=False)  # Randomly select 30 % of samples
        projected_features = TSNE(n_components=2, random_state=0).fit_transform(
            features[indices])
        my_id = 0

        for category, category_id in sorted(category_to_id.items()):
            points = projected_features[(
                labels[indices] == category_id).values]
        models = [
            RandomForestClassifier(
                n_estimators=200, max_depth=100, random_state=0),
            MultinomialNB(),
            LogisticRegression(random_state=30),
        ]
        CV = 5
        cv_df = pd.DataFrame(index=range(CV * len(models)))
        entries = []
        for model in models:
            model_name = model.__class__.__name__
            accuracies = cross_val_score(
                model, features, labels, scoring='accuracy', cv=CV)
            for fold_idx, accuracy in enumerate(accuracies):
                entries.append((model_name, fold_idx, accuracy))
            cv_df = pd.DataFrame(
                entries, columns=['model_name', 'fold_idx', 'accuracy'])

        model = LogisticRegression(random_state=0)

        # Split Data
        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index,
                                                                                         test_size=0.33,
                                                                                         random_state=42)

        # Train Algorithm
        model.fit(X_train, y_train)


        # Make Predictions
        y_pred_proba = model.predict_proba(X_test)
        y_pred = model.predict(X_test)
        filename = 'finalized_model.sav'
        pickle.dump(model, open(filename, 'wb'))

def retrainModel(docs):
    filename = 'finalized_model.sav'
    model = pickle.load(open(filename, 'rb'))
    docs['category_id'] = docs['category'].factorize()[0]
    category_id_df = docs[['category', 'category_id']
                        ].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'category']].values)
    docs.groupby('category').category_id.count()
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                            stop_words='english')

    features = tfidf.fit_transform(docs.description.values.astype('U')).toarray()
    labels = docs.category_id
    N = 3
    for Category, category_id in sorted(category_to_id.items()):
        features_chi2 = chi2(features, labels == category_id)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        SAMPLE_SIZE = int(len(features) * 0.3)
        np.random.seed(0)
        indices = np.random.choice(range(len(features)), size=SAMPLE_SIZE,
                                   replace=False)  # Randomly select 30 % of samples
        projected_features = TSNE(n_components=2, random_state=0).fit_transform(
            features[indices])
        my_id = 0
        for category, category_id in sorted(category_to_id.items()):
            points = projected_features[(
                labels[indices] == category_id).values]
        models = [
            RandomForestClassifier(
                n_estimators=200, max_depth=100, random_state=0),
            MultinomialNB(),
            LogisticRegression(random_state=30),
        ]

        # Split Data
        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, docs.index,
                                                                                         test_size=0.33,
                                                                                         random_state=42)



        # Make Predictions
        y_pred = model.partial_fit(X_test)

        N = 5
        for Category, category_id in sorted(category_to_id.items()):
            # This time using the model co-eficients / weights
            indices = np.argsort(model.coef_[category_id])
            feature_names = np.array(tfidf.get_feature_names())[indices]
            unigrams = [v for v in reversed(
                feature_names) if len(v.split(' ')) == 1][:N]
            bigrams = [v for v in reversed(
                feature_names) if len(v.split(' ')) == 2][:N]
            #print("# '{}':".format(Category))
            #print("  . Top unigrams:\n       . {}".format('\n       . '.join(unigrams)))
            #print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))
            test_features = tfidf.transform(docs.description.tolist())

            Y_pred = model.predict(test_features)
            print("predictions")
            Y_pred_category = []
            for cat in Y_pred:
                Y_pred_category = id_to_category[cat]
            predictions = pd.DataFrame({
                "ArticleId": docs["_id"],
                "Description": docs["description"],
                "Category": Y_pred_category
            })
            predictions.to_csv('predictions.csv', index=False)
            print(predictions)


if __name__ == "__main__":
    docs=convert_data_to_csv()
    trainModel()
    retrainModel( docs)





def predict():
    print("end")