#!/usr/bin/env python


import argparse
import csv
import json
import sys
import time
from confluent_kafka import Producer
import socket
from newsapi import NewsApiClient
import http.client
import urllib.parse
import pandas as pd
import numpy as np

offset = 0


def acked(err, msg):
    if err is not None:
        print("Failed to deliver message: %s: %s" %
              (str(msg.value()), str(err)))
    else:
        print("Message produced: %s" % (str(msg.value())))


def main():
    counter = 0
    current_article_count = None
    parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument('filename', type=str,
    # help='Time series csv file.')
    parser.add_argument('topic', type=str,
                        help='Name of the Kafka topic to stream.')
    # parser.add_argument('--speed', type=float, default=1, required=False,
    #                     help='Speed up time series by a given multiplicative factor.')
    args = parser.parse_args()

    topic = args.topic
    p_key1 = "newsapi"
    p_key2 = "mediastack"
    conf = {'bootstrap.servers': "localhost:9092",
            'client.id': socket.gethostname()}
    producer = Producer(conf)
    running = True
    while running:
        try:
            # mediastack news
            data = combine_cat_data()

            print("API time break.....")

            for article in data:
                payload = {
                    'title': article["title"],
                    "category": article["category"],
                    "description": article["description"]
                }
                payload = json.dumps(payload)
                producer.produce(topic=topic, key=p_key2,
                                 value=payload, callback=acked)

            producer.flush()
            time.sleep(90)  # temp change

        except Exception as e:
            if e == TypeError:
                sys.exit()
            else:
                print(e)


def get_mediastack():
    global offset
    conn = http.client.HTTPConnection('api.mediastack.com')
    params = urllib.parse.urlencode({
        'access_key': '85b48d9edcb0a2a1d38c7e0ac0eb8919',  # ysusheen api key
        # 'categories': '-general,-sports,-bussiness,-entertainment,-health,-science,-technology',
        'sort': 'published_desc',
        'language': "en,-ar,-de,-es,-fr,-he,-it,-nl,-no,-pt,-ru,-se,-zh",
        'limit': 100,
    })

    conn.request('GET', '/v1/news?{}'.format(params))

    res = conn.getresponse()
    data = res.read().decode("utf-8")
    data = json.loads(data)

    articles = data["data"]
    articles = list(
        filter(
            lambda article: True if article["language"] == 'en' else False, articles))
    return articles


def get_balanced_mediastack(category, offset=0):
    conn = http.client.HTTPConnection('api.mediastack.com')
    params = urllib.parse.urlencode({
        'access_key': '85b48d9edcb0a2a1d38c7e0ac0eb8919',
        'sort': 'published_desc',
        'language': "en,-ar,-de,-es,-fr,-he,-it,-nl,-no,-pt,-ru,-se,-zh",
        'categories': category,
        'offset': offset,
        'limit': 50,
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
        print("Error", e)


def combine_cat_data():
    new_data = True
    data = None
    categories = ["general", 'technology', 'business',
                  'science', 'sports', 'health', 'entertainment']

    for category in categories:
        articles_data = get_balanced_mediastack(category, offset)
        if new_data:
            data = articles_data
            new_data = False
        else:
            data = pd.concat([data, articles_data])

    data = data.sample(frac=1)
    articles = data.to_dict('records')
    return articles


if __name__ == "__main__":
    main()
