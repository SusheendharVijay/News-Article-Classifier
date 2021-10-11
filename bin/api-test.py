from newsapi import NewsApiClient
import json


def main():
    # Init
    newsapi = NewsApiClient(api_key='4ddbf382b16c4184a33bdd8453be9a42')

    # /v2/top-headlines
    news = newsapi.get_everything(q='bitcoin')
    print(news)


if __name__ == '__main__':
    main()
