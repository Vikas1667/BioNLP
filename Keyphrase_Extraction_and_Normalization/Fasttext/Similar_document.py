from gensim.models import word2vec
from gensim.models.fasttext import FastText
from gensim.models import Word2Vec
from gensim.test.utils import datapath
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import nltk
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk import WordNetLemmatizer, pos_tag, word_tokenize

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

stop_words = stopwords.words('english')

#####################################################################################
pat_data = pd.read_excel('../data/recent_5k.xlsx')
abs_text = pat_data['Abstract']
abs_list = pat_data['Abstract'].tolist()


#####################################################################################
# Tokenizing with simple preprocess gensim's simple preprocess

def sent_to_words(sentences):
    for sentence in sentences:
        yield (simple_preprocess(str(sentence),
                                 deacc=True))  # returns lowercase tokens, ignoring tokens that are too short or too long


def remove_stopwords(sentence):
    filtered_words = [word for word in sentence if word not in stop_words]
    return filtered_words


abs_words = list(sent_to_words(abs_list))
lengths = [len(abst) for abst in abs_words]
plt.hist(lengths, bins=25)
plt.show()

filtered_abst = [remove_stopwords(abst) for abst in abs_words]
lengths = [len(abst) for abst in filtered_abst]
plt.hist(lengths, bins=25)
plt.show()

print('Mean word count of abs is %s' % np.mean(lengths))

#################################################################################
n = 50
ft_model = FastText(filtered_abst, vector_size=n, window=8, min_count=5, workers=2, sg=1)

#####################################################################################


# To proprely work with scikit's vectorizer
merged_questions = [' '.join(question) for question in filtered_abst]
document_names = ['Doc {:d}'.format(i) for i in range(len(merged_questions))]


def get_tfidf(docs, ngram_range=(1, 1), index=None):
    vect = TfidfVectorizer(stop_words='english', ngram_range=ngram_range)
    tfidf = vect.fit_transform(docs).todense()
    return pd.DataFrame(tfidf, columns=vect.get_feature_names(), index=index).T


tfidf = get_tfidf(merged_questions, ngram_range=(1, 1), index=document_names)


def get_sent_embs(emb_model):
    sent_embs = []
    for desc in range(len(filtered_abst)):
        sent_emb = np.zeros((1, n))
        if len(filtered_abst[desc]) > 0:
            sent_emb = np.zeros((1, n))
            div = 0
            model = emb_model
            for word in filtered_abst[desc]:
                if word in model.wv.key_to_index and word in tfidf.index:
                    word_emb = model.wv[word]
                    weight = tfidf.loc[word, 'Doc {:d}'.format(desc)]

                    sent_emb = np.add(sent_emb, word_emb * weight)
                    div += weight
                else:
                    div += 1e-13  # to avoid dividing by 0
        if div == 0:
            print(desc)

        sent_emb = np.divide(sent_emb, div)
        sent_embs.append(sent_emb.flatten())
    return sent_embs


ft_sent = get_sent_embs(emb_model=ft_model)


######################################################################
def get_n_most_similar(interest_index, embeddings, n):
    """
    Takes the embedding vector of interest, the list with all embeddings, and the number of similar abstract to
    retrieve.
    Outputs the disctionary IDs and distances
    """
    nbrs = NearestNeighbors(n_neighbors=n, metric='cosine').fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    similar_indices = indices[interest_index][1:]
    similar_distances = distances[interest_index][1:]
    return similar_indices, similar_distances


def print_similar(interest_index, embeddings, n):
    """
    Convenience function for visual analysis
    """
    closest_ind, closest_dist = get_n_most_similar(interest_index, embeddings, n)
    print('Abstract %s \n \n is most similar to these %s abstract: \n' % (abs_list[interest_index], n))
    for question in closest_ind:
        print('ID ', question, ': ', abs_list[question])


print_similar(42, ft_sent, 5)

#################################################################################

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(filtered_abst)]
model = Doc2Vec(documents, vector_size=n, window=8, min_count=5, workers=2, dm=1, epochs=20)
print(abs_list[42], ' \nis similar to \n')
print([abs_list[similar[0]] for similar in model.docvecs.most_similar(42)])
