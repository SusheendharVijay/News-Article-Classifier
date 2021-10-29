# get some libraries that will be useful
import re
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# the Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
# function to split the data for cross-validation
from sklearn.model_selection import train_test_split
# function for transforming documents into counts
from sklearn.feature_extraction.text import CountVectorizer
# function for encoding categories
from sklearn.preprocessing import LabelEncoder


def normalize_text(s):
    s = str(s)
    s = s.lower()

    # remove punctuation that is not word-internal (e.g., hyphens, apostrophes)
    s = re.sub('\s\W', ' ', s)
    s = re.sub('\W\s', ' ', s)

    # make sure we didn't introduce any double spaces
    s = re.sub('\s+', ' ', s)

    return s


# grab the data


news = pd.read_csv("combined.csv")
for col in ['title', 'description']:
    news[col] = [normalize_text(s) for s in news[col]]

# pull the data into vectors
vec1 = CountVectorizer()
vec2 = CountVectorizer()
# x1 = vec1.fit_transform(news['title'])
x2 = vec2.fit_transform(news['description'])

# X = np.concatenate((x1, x2), axis=1)
encoder = LabelEncoder()
y = encoder.fit_transform(news['category'])

# split into train and test sets
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print(x2.shape)
# print(x1.shape)


print(x1)
