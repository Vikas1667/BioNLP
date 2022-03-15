import os
import pandas as pd
import numpy as np
from torch import nn
# import umap
# import hdbscan
from tqdm import tqdm

import spacy
nlp = spacy.load("en_core_sci_sm")

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, models

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
stop_words=stopwords.words('english')
from transformers import logging
# from code.ctfidf import CTFIDFVectorizer

logging.set_verbosity_warning()



"""Storing and Loading model in Pretrain_model directory"""
cwd = os.getcwd()
# model_dir = './Pretrain_model'

"""Bert Model for Sentence Embedding for each abstract """

# word_embedding_model = models.Transformer('distilbert-base-uncased', max_seq_length=128)
# pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension()).to('cuda:0')
# dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=64,
#                            activation_function=nn.Tanh()).to('cuda:0')
# sent_model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model]).to('cuda:0')


sent_model = models.Transformer('gsarti/scibert-nli', max_seq_length=256)




def get_feature_word(text, model):
    """Index2word is a list that contains the names of the words inthe models vocabulary."""
    clean_word_list = []

    index2word_set = set(model.index_to_key)
    tokens = text.split()
    for word in tokens:
        if word in index2word_set:
            clean_word_list.append(word)
    return clean_word_list


def candidate_keywords(text, model):
    """BioWordVec model & word2vec with Fasttext for candidate selection"""
    candidate_keys = get_feature_word(text, model)
    return candidate_keys


def scispacy_cand_keys(text):
    doc=nlp(str(text))
    cand_keys = [e.text for e in doc.ents]
    return cand_keys


def tfidf_candidate_keywords(text):
    tfidf_countvec = TfidfVectorizer(stop_words=stop_words, min_df=0.02).fit([text])
    tfidf_words = tfidf_countvec.get_feature_names()
    return tfidf_words


def mmr(doc_embedding: np.ndarray, word_embeddings: np.ndarray, words, top_n, diversity):
    """Maximal Marginal Retrieval"""

    # Extract similarity within words, and between words and the document
    try:
        word_similarity = cosine_similarity(word_embeddings)
        word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)

        # Initialize candidates and already choose best keyword/keyphras
        keywords_idx = [np.argmax(word_doc_similarity)]
        candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

        for _ in range(top_n - 1):
            # Extract similarities within candidates and
            # between candidates and selected keywords/phrases
            candidate_similarities = word_doc_similarity[candidates_idx, :]
            target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

           # Calculate MMR
            mmr = (1 - diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
            if mmr.size > 0:
                mmr_idx = candidates_idx[np.argmax(mmr)]
                # Update keywords & candidates
                keywords_idx.append(mmr_idx)
                candidates_idx.remove(mmr_idx)

        return [(words[idx], round((word_doc_similarity.reshape(1, -1)[0][idx]), 4)) for idx in keywords_idx]

    except ValueError:
        pass

def word_doc_embedding(text,candidate_keys):
    word_embedding = sent_model.encode(candidate_keys)
    doc_embedding = sent_model.encode([text])
    return word_embedding,doc_embedding


def Keyword_extractor(text, cand_keys):
    keywords = []
    try:
        word_emb, doc_emb = word_doc_embedding(text, cand_keys)
        keyword_doc = mmr(doc_emb, word_emb, cand_keys, 20, 0.8)
        keyword_doc = ','.join([key[0] for key in keyword_doc])
        keywords.append(keyword_doc)
        return keywords
    except TypeError:
        pass


# def ctfidf(corpus, ngram_range=(1, 1)):
#
#     model = SentenceTransformer('distilbert-base-nli-mean-tokens')
#     embedding = model.encode(corpus, show_progress_bar=True)
#     umap_embeddings = umap.UMAP(n_neighbors=15,
#                                 n_components=5,
#                                 metric='cosine').fit_transform(embedding)
#
#     cluster = hdbscan.HDBSCAN(min_cluster_size=15,
#                               metric='euclidean',
#                               cluster_selection_method='eom').fit(umap_embeddings)
#
#     docs_df = pd.DataFrame(corpus, columns=["Doc"])
#     docs_df['Topic'] = cluster.labels_
#     docs_df['Doc_ID'] = range(len(docs_df))
#     docs_per_topic = docs_df.groupby(['Topic'], as_index=False).agg({'Doc': ' '.join})
#     count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(docs_per_topic.Doc.values)
#     ctfidf = CTFIDFVectorizer().fit_transform(count)

