#!/usr/bin/env python

"""Generates a stream to Kafka from a time series csv file.
"""

import argparse
import csv
import json
import sys
import time
from confluent_kafka import Producer
import socket
from newsapi import NewsApiClient


def acked(err, msg):
    if err is not None:
        print("Failed to deliver message: %s: %s" %
              (str(msg.value()), str(err)))
    else:
        print("Message produced: %s" % (str(msg.value())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument('filename', type=str,
    # help='Time series csv file.')
    parser.add_argument('topic', type=str,
                        help='Name of the Kafka topic to stream.')
    # parser.add_argument('--speed', type=float, default=1, required=False,
    #                     help='Speed up time series by a given multiplicative factor.')
    args = parser.parse_args()

    topic = args.topic
    p_key = "newsapi"

    conf = {'bootstrap.servers': "localhost:9092",
            'client.id': socket.gethostname()}
    producer = Producer(conf)
    running = True
    while running:
        try:
            newsapi = NewsApiClient(api_key='4ddbf382b16c4184a33bdd8453be9a42')

            news = newsapi.get_top_headlines()
            print("streaming news")
            payload = json.dumps(news)
            producer.produce(topic=topic, key=p_key,
                             value=payload, callback=acked)
            producer.flush()
            time.sleep(10)
        except TypeError:
            sys.exit()


if __name__ == "__main__":
    main()
