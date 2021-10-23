import http.client
import urllib.parse
import json

conn = http.client.HTTPConnection('api.mediastack.com')

params = urllib.parse.urlencode({
    'access_key': 'db473f12969c8297f6a4453ca4ebd5d5',
    # 'categories': '-general,-sports,-bussiness,-entertainment,-health,-science,-technology',
    'sort': 'published_desc',
    'limit': 10,
})

conn.request('GET', '/v1/news?{}'.format(params))

res = conn.getresponse()
data = res.read().decode('utf-8')

print(json.loads(data))
