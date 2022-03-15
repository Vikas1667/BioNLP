import re
from string import punctuation
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer, word_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import nltk
nltk.download('wordnet')

lemma = WordNetLemmatizer()
stop_words=stopwords.words('english')

# Extraction of DataFrame
'''In Bootstrap we are using 5K records of SOLR records for NLP-Keyword Tagging task'''

# def text_cleaning(text):
#     cleaned_txt = re.sub('<[^<]+>', '', str(text))
#     cleaned_txt = re.sub(r'\[.*?\]', '', cleaned_txt)
#     cleaned_txt = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned_txt)
#     tokens = re.split(r'\s+', cleaned_text.lower().strip())

#     return cleaned_txt


# def StopWord_removal(text):
#     cleaned_txt = ' '.join([token for token in word_tokenize(text) if token not in stop_words])
#     return cleaned_txt


# def Lemma(text):
#     cleaned_txt = ' '.join([lemma.lemmatize(token) for token in word_tokenize(text)])
#     return cleaned_txt

def normalize_text(text):
    # remove special characters\whitespaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text), re.I | re.A)

    # lower case & tokenize text
    tokens = word_tokenize(text)

    # filter stopwords out of text
    stop_word = set(stopwords.words('english')) | set(punctuation)| set(ENGLISH_STOP_WORDS)

    # re-create text from filtered lemmatize tokens
    cleaned_text = ' '.join(lemma.lemmatize(token) for token in tokens if token not in stop_word)

    return cleaned_text

