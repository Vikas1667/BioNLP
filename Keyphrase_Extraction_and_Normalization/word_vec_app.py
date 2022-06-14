from gensim.models import Word2Vec
from flask import Flask, jsonify, request
from gensim.models import FastText, KeyedVectors
import json

app = Flask(__name__)


@app.route('/most_similar',methods=['GET'])
def word2vec():
    model = Word2Vec.load('word_embedding/word2vec5k.bin')
    w = request.args.get('word')
    return json.dumps(model.wv.most_similar(positive=[w], topn=10))


@app.route('/bioword_most_similar',methods=['GET'])
def bio_word2vec():
    model = KeyedVectors.load_word2vec_format('./word_embedding/bio_embedding_intrinsic.bin',binary=True)
    print(model)
    w = request.args.get('word')
    return json.dumps(model.most_similar(positive=[w], topn=10))


if __name__ == '__main__':
    app.run(debug=True)
