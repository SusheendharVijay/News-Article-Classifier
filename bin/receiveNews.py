#!/usr/bin/env python

"""Consumes stream for printing all messages to the console.
"""

import argparse
import json
import sys
import time
import socket
from confluent_kafka import Consumer, KafkaError, KafkaException
from pymongo import MongoClient


def msg_process(msg):

    try:
        dmsg = json.loads(msg.value())
        print(dmsg["description"])
    except:
        print("Empty data received")
        return
    try:
        conn = MongoClient('mongodb://root:example@localhost:27017', 27017)
        # print("connected sucessfully")
    except:
        print("could not connect to mongodb")
        return

    db = conn.database
    users = db.users
    # emp_rec1 = {
    #     "name": "Mr.Geek",
    #     "eid": 24,
    #     "location": "delhi"
    # }
    # emp_rec2 = {
    #     "name": "Mr.Shaurya",
    #     "eid": 14,
    #     "location": "delhi"
    # }
    # rec_id1 = users.insert_one(emp_rec1)
    # rec_id2 = users.insert_one(emp_rec2)
    # print("Data inserted with record ids", rec_id1, " ", rec_id2)
    news_rec = {
        "source": dmsg['source'],
        "description": dmsg['description'],
        "title": dmsg["title"],
        "category": dmsg["category"]
    }
    rec_id = users.insert_one(news_rec)
    print("record entered with rec_id:", rec_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('topic', type=str,
                        help='Name of the Kafka topic to stream.')

    args = parser.parse_args()

    conf = {'bootstrap.servers': 'localhost:9092',
            'default.topic.config': {'auto.offset.reset': 'smallest'},
            'group.id': socket.gethostname()}

    consumer = Consumer(conf)

    running = True

    try:
        while running:
            consumer.subscribe([args.topic])

            msg = consumer.poll(1)
            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # End of partition event
                    sys.stderr.write('%% %s [%d] reached end at offset %d\n' %
                                     (msg.topic(), msg.partition(), msg.offset()))
                elif msg.error().code() == KafkaError.UNKNOWN_TOPIC_OR_PART:
                    sys.stderr.write('Topic unknown, creating %s topic\n' %
                                     (args.topic))
                elif msg.error():
                    raise KafkaException(msg.error())
            else:
                msg_process(msg)

    except KeyboardInterrupt:
        pass

    finally:
        # Close down consumer to commit final offsets.
        consumer.close()


if __name__ == "__main__":
    main()
