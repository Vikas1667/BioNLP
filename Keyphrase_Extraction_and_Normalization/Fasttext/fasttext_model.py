from gensim.models.fasttext import FastText
import pandas as pd
import re
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk import WordNetLemmatizer, pos_tag, word_tokenize
from tqdm import tqdm
from gensim.models import word2vec
from gensim.models.fasttext import FastText
from gensim.models import Word2Vec
from gensim.test.utils import datapath

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
stemmer = WordNetLemmatizer()


def preprocess_text(document):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(document))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    tokens = document.split()
    tokens = [stemmer.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if len(word) > 3]

    preprocessed_text = ' '.join(tokens)

    return preprocessed_text


def word_corpus_builder(sents_list):
    final_corpus = [preprocess_text(sentence) for sentence in sents_list if str(sentence).strip() != '']
    print(final_corpus[1])
    word_punctuation_tokenizer = nltk.WordPunctTokenizer()
    word_tokenized_corpus = [word_punctuation_tokenizer.tokenize(sent) for sent in final_corpus]
    print(word_tokenized_corpus[10])
    return word_tokenized_corpus


def fasttext_model(word_tokenized_corpus):
    embedding_size = 60
    window_size = 40
    min_word = 5
    down_sampling = 1e-2

    ft_model = FastText(word_tokenized_corpus,
                        vector_size=embedding_size,
                        window=window_size,
                        min_count=min_word,
                        sample=down_sampling,
                        sg=1)

    ft_model.build_vocab(word_tokenized_corpus)
    total_words = ft_model.corpus_total_words
    ft_model.train(word_tokenized_corpus, total_words=total_words, epochs=5)
    print(ft_model.wv.similarity('crispr', 'crisprcas'))
    print(ft_model.wv.similar_by_word('crispr'))
    return ft_model


def get_feature_vec_fast(text, model):
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary.
    clean_text = []
    index2word_set = set(model.wv.index_to_key)
    words = set(text.split())

    for word in words:
        if word in index2word_set:
            clean_text.append(model[word])
    print('WORD2vec text voc len', len(clean_text))
    return clean_text


if __name__ == "__main__":
    sample_data = pd.read_excel('../data/recent_5k.xlsx')
    sample_abs_data = sample_data['Abstract']
    sents_list = sample_abs_data.to_list()

    word_tokenized_corpus = word_corpus_builder(sents_list)
    ft_model = fasttext_model(word_tokenized_corpus)
    final_corpus = [preprocess_text(sentence) for sentence in sents_list if str(sentence).strip() != '']
    print(final_corpus[1])
    for i in final_corpus:
        trainDataVecs_fast = get_feature_vec_fast(i, ft_model)
        print(trainDataVecs_fast)
        break

    ft_model.save('../word_embedding/word2vec5k.bin')
    # model=Word2Vec.load('V:/ML_projects/Merckgroup/Projects/Keyword_tagging/Pretrain_model/word2vec5k.bin')
