import math

from lib.search_utils import (
    load_movies,
    load_stopwords,
    CACHE_PATH,
    BM25_K1,
    BM25_B,
)
import string
from nltk.stem import PorterStemmer as stemmer
from collections import Counter, defaultdict
import os
import pickle

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)

        self.index_path = CACHE_PATH/'index.pkl'
        self.docmap_path = CACHE_PATH/'docmap.pkl'
        self.term_frequencies_path = CACHE_PATH/'term_frequencies.pkl'
        self.doc_lengths = {}
        self.doc_lengths_path = CACHE_PATH/'doc_lengths.pkl'

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)

    def get_documnets(self, term):
        return sorted(list(self.index[term]))

    def get_tf(self, doc_id, term):
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("can only have 1 tokens")
        return self.term_frequencies[doc_id][tokens[0]]
        
    def get_idf(self, term):
        token = tokenize_text(term)
        if len(token) != 1:
            raise ValueError("can only have 1 tokens")
        token = token[0]
        doc_count = len(self.docmap)
        term_match_doc_count = len(self.index[token])
        return math.log((doc_count + 1) / (term_match_doc_count +1))

    def get_tfidf(self, doc_id, term):
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf
    
    def get_bm25_idf(self, term: str) -> float:
        token = tokenize_text(term)
        if len(token) != 1:
            raise ValueError("can only have 1 tokens")
        token = token[0]
        N=len(self.docmap)
        df = len(self.index[token])
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie['id']
            title = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, title)
            self.docmap[doc_id] = movie

    def save(self):
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, 'wb') as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, 'wb') as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, 'wb') as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        with open(self.index_path, 'rb') as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, 'rb') as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, 'rb') as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, 'rb') as f:
            self.doc_lengths = pickle.load(f)
    
    def __get_avg_doc_length(self) -> float:
        lenghts= list(self.doc_lengths.values())
        if len(lenghts) == 0:
            return 0.0
        ttl = 0
        for l in lenghts:
            ttl += l
        return ttl / len(lenghts)
        
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        if avg_doc_length > 0:
            length_norm = 1 - b + b * (doc_length / avg_doc_length)
        else:
            length_norm = 1
        tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return tf_component
    
    def bm25(self, doc_id, term):
        bm25tf = self.get_bm25_tf(doc_id, term)
        bm25idf = self.get_bm25_idf(term)
        return bm25tf * bm25idf
    
    def bm25_search(self, query, limit=5):
        tokens = tokenize_text(query)
        scrors = {}
        for doc_id in self.docmap:
            scrose = 0
            for token in tokens:
                scrose += self.bm25(doc_id, token)
                scrors[doc_id] = scrose
        def _key(x): return x[1]
        sorted_scores = sorted(scrors.items(), key=lambda x: x[1], reverse=True)
        result = sorted_scores[:limit]
        formate_result = []
        for doc_id, score in result:
            title = self.docmap[doc_id]
            formate_result.append(
                {
                    "doc_id": doc_id,
                    "title": title["title"],
                    "score": score
                }
            )
        return formate_result


def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()

def tf_command(doc_id, term):
    idx = InvertedIndex()
    idx.load()
    print(idx.get_tf(doc_id, term))

def tfidf_command(doc_id, term):
    idx = InvertedIndex()
    idx.load()
    tf_idf = idx.get_tfidf(doc_id, term)
    print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")

def bm25_idf_command(term):
    idx = InvertedIndex()
    idx.load()
    bm25idf = idx.get_bm25_idf(term)
    print(f"BM25 IDF score of '{term}': {bm25idf:.2f}")

def idf_command(term):
    idx = InvertedIndex()
    idx.load()
    idf = idx.get_idf(term)
    print(f"Inverse document frequency of '{term}': {idf:.2f}")

def bm25_tf_command(docs_id, term, k1=BM25_K1, b=BM25_B):
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(docs_id, term, k1, b) 

def bm25_search(query):
    idx = InvertedIndex()
    idx.load()
    return idx.bm25_search(query, limit=5)


def clean_text(text) :
    text = text.lower()
    text = text.translate(str.maketrans("","",string.punctuation))
    return text
 
def tokenize_text(text):
    text = clean_text(text)
    stopwords = load_stopwords()
    res = []
    def _filter(tok):
        if tok and tok not in stopwords:
            return True
        return False
    for tok in text.split():
        if _filter(tok):
            tok = stemmer().stem(tok)
            res.append(tok)
    return res

def has_matching_token(query_tokens, movie_tokens) :
    for query_tok in query_tokens :
        for movie_tok in movie_tokens :
            if query_tok in movie_tok :
                return True
    return False

def search_command(query, n_results=5) :
    movies = load_movies()
    idx = InvertedIndex() 
    idx.load()
    seen, res = set(), []
    query_tokens = tokenize_text(query)
    for qt in query_tokens :
        matching_docs = idx.get_documnets(qt)
        for matching_doc_id in matching_docs :
            if matching_doc_id in seen :
                continue
            seen.add(matching_doc_id)
            doc = idx.docmap[matching_doc_id]
            res.append(doc)

            if len(res) >= n_results :
                return res
    return res 