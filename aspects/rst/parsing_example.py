import json

import nltk
import requests

if __name__ == '__main__':
    payload = {
        'text': 'I love this phone. It is the best screen I have ever seen.'
    }
    r = requests.post('http://oxygen.engine.kdm.wcss.pl:5000/api/rst/parse', json=payload)
    # tree = nltk.Tree.fromstring(r.text)
    tree = nltk.Tree.fromstring(json.loads(r.text)['tree'])
