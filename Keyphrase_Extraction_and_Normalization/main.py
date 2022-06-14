
from Preprocessing_text import normalize_text
import pandas as pd
from tqdm import tqdm
# from keyword_extractor import tfidf_candidate_keywords
from gensim.models import FastText, KeyedVectors

from keyword_extractor import Keyword_extractor,scispacy_cand_keys,candidate_keywords
from keyword_extractor import word_doc_embedding


if __name__ == "__main__":
    """BioWordVec model"""
    BioWordVec_path = './word_embedding/bio_embedding_intrinsic.bin'
    BioWordVec_model = KeyedVectors.load_word2vec_format(BioWordVec_path, binary=True)

    text_df = pd.read_excel('./data/recent_5k.xlsx',engine='openpyxl')
    clean_abs_text = text_df['Abstract'].apply(normalize_text)
    text_docs = clean_abs_text[:100].tolist()
    keywords_list = []


    for text_doc in tqdm(text_docs):
            # candKeys = tfidf_candidate_keywords(text_doc)

            sci_candKeys=scispacy_cand_keys(text_doc)
            bioword_candKeys= candidate_keywords(text_doc, BioWordVec_model)

            candKeys=sci_candKeys+bioword_candKeys
            # print(candKeys)
            word_embed, doc_embed = word_doc_embedding(text_doc,candKeys)
            keywords_doc = Keyword_extractor(text_doc,candKeys)
            print(keywords_doc)

            keywords_list.append(keywords_doc)

    keywords_df=pd.DataFrame()
    keywords_df['Abstract']=text_docs
    keywords_df['Keywords_extracted']=keywords_list

    # keywords_df.to_excel('./data/recent5k_output_27-10.xlsx')


