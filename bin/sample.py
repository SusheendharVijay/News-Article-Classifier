from pymongo import MongoClient, database
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

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
    #print(mongo_docs)
    docs = pd.DataFrame(columns=[])
    for num, doc in enumerate( mongo_docs ):
        doc["_id"] = str(doc["_id"])
        doc_id = doc["_id"]
        series_obj = pd.Series( doc, name=doc_id )
        docs = docs.append( series_obj )
    
    docs.to_csv("news_data.csv", ",")
    csv_export = docs.to_csv(sep=",")
    print ("\nCSV data:***********************", csv_export)
    data = pd.read_csv("news_data.csv", sep='\t')
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``")
    print(docs)

    x = np.array(docs["title"])
    y = np.array(docs["category"])

    cv = CountVectorizer()
    X = cv.fit_transform(x)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    print(X_train)
    print(y_train)
    model = MultinomialNB()
    model.fit(X_train,y_train)
    user = input("Enter a Text: ")
    data = cv.transform([user]).toarray()
    output = model.predict(data)
    print(output)  

    

if __name__ == "__main__":
    convert_data_to_csv()
