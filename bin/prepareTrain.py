import http.client
import urllib.parse
import json
import pandas as pd
import numpy as np


def get_mediastack(category, offset):

    conn = http.client.HTTPConnection('api.mediastack.com')
    params = urllib.parse.urlencode({
        'access_key': 'db473f12969c8297f6a4453ca4ebd5d5',
        # 'categories': '-general,-sports,-bussiness,-entertainment,-health,-science,-technology',
        'sort': 'published_desc',
        'language': "en,-ar,-de,-es,-fr,-he,-it,-nl,-no,-pt,-ru,-se,-zh",
        'categories': category,
        'offset': offset,
        'limit': 100,
    })
    try:
        conn.request('GET', '/v1/news?{}'.format(params))

        res = conn.getresponse()
        data = res.read().decode("utf-8")
        data = json.loads(data)

        articles = data["data"]
        articles = list(
            filter(
                lambda article: True if article["language"] == 'en' else False, articles))
        articles = map(lambda article: {
            'title': article["title"],
            "category": article["category"],

            "description": article["description"]
        }, articles)
        return pd.DataFrame(articles)

    except Exception as e:
        print(e)


def combine_cat_data():
    new_data = True
    data = None
    categories = ["general", 'technology', 'business',
                  'science', 'sports', 'health', 'entertainment']
    for offset in [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        print(offset)
        for category in categories:
            articles_data = get_mediastack(category, offset)
            if new_data:
                data = articles_data
                new_data = False
            else:
                data = pd.concat([data, articles_data])
    return data


data = combine_cat_data()
data.to_csv('combined.csv', index=False)
