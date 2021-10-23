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
            # newsapi = NewsApiClient(api_key='4ddbf382b16c4184a33bdd8453be9a42')
            # news = newsapi.get_top_headlines()
            # article_count = len(news["articles"])
            # if current_article_count == None:
            #     current_article_count = article_count
            # elif article_count != current_article_count:
            #     counter = 0
            #     current_article_count = article_count
            # else:
            #     counter += 1

            # if counter == current_article_count:
            #     print("No more articles, returning...")
            #     return

            # articles = news["articles"]
            # print("stats - counter:{}, current_article_count:{}, api article_count:{}".format(counter,
            #                                                                                   current_article_count, article_count))
            # payload = json.dumps(articles[counter])
            # producer.produce(topic=topic, key=p_key,
                            #  value=payload, callback=acked)
            # producer.flush()

            # mediastack news
            data = get_mediastack()
            time.sleep(120)
            for article in data:
                payload = {
                    'title': article["title"],
                    "category": article["category"],
                    "source": article["source"],
                    "description": article["description"]
                }
                payload = json.dumps(payload)
                producer.produce(topic=topic, key=p_key2,
                                 value=payload, callback=acked)

            producer.flush()

        except TypeError:
            sys.exit()


def get_mediastack():
    global offset
    conn = http.client.HTTPConnection('api.mediastack.com')
    params = urllib.parse.urlencode({
        'access_key': 'db473f12969c8297f6a4453ca4ebd5d5',
        # 'categories': '-general,-sports,-bussiness,-entertainment,-health,-science,-technology',
        'sort': 'published_desc',
        'language': "en",
        'limit': 30,
    })

    conn.request('GET', '/v1/news?{}'.format(params))

    res = conn.getresponse()
    data = res.read().decode("utf-8")
    data = json.loads(data)

    return data["data"]


if __name__ == "__main__":
    main()
