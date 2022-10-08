from typing import Union
from fastapi import FastAPI
from gensim.models import KeyedVectors
from gensim import models
import numpy as np
import sys
sys.setrecursionlimit(1500)

# import request
app = FastAPI()

word2vec_bin = './pubmed_mesh_test.bin'
w2v_model = models.KeyedVectors.load_word2vec_format(word2vec_bin, binary=True)


def bioword_similarity(word1, word2):
    return float(w2v_model.similarity(word1, word2))


@app.get("/")
def welcome():
    return "Welcome to our Machine Learning REST API!"


@app.get("/similarity")
def similarity(word1: str, word2: str):
    wrd_sim=bioword_similarity(word1, word2)
    # print(wrd_sim)

    return {'word similarity': wrd_sim}

# if __name__ == '__main__':
#     similarity('diabetes','diabetic')
