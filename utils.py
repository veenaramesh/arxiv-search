import numpy as np
import pickle
import spacy
from tqdm import tqdm
from gensim.models.fasttext import FastText
from rank_bm25 import BM25Okapi
import nmslib

def tokenize(dataframe_column, nlp=spacy.load("en_core_web_md")): 
    """
    PARAMETERS
    ----------
    dataframe_column: pandas series or similar type to tokenize
    nlp: spacy pipeline for CPU (default is the en_core_web_md)

    RETURNS
    -------
    list of tokenized words from dataframe_column
    """
    tokens =[] 

    text = dataframe_column.str.lower().values
    text = [str(i) for i in text]

    for doc in tqdm(nlp.pipe(text, n_threads=2, disable=["tagger", "parser","ner"])):
        tok = [t.text for t in doc if (t.is_ascii and not t.is_punct and not t.is_space)]
        tokens.append(tok)

    return tokens 

def creatingModel(tokens, skipgram=1, size=100, window=10, minimum_count=5, negative_subsampling=15, minimum_n=2, maximum_n=5, epochs=6, save=True, model_name="models/_fasttext.model"): 
    """
    PARAMETERS
    ----------
    tokens: tokenized list of words
    skipgram: for fastText model; use skip-gram: usually gives better results
    size: for fastText model; embedding dimension (default)
    window: for fastText model; the context for the model essentially
    minimum_count: for fastText model; model only considers tokens with at least n occurrences in the corpus
    negative_subsampling: for fastText model; samples negative examples
    minimum_n: for fastText model; min character n-gram
    maximum_n: for fastText model; max character n-gram
    epochs: the number of epochs when training fastText model
    save: option to save the fastText model
    model_name: if "save" is True, saves the model as the model_name 

    RETURNS
    -------
    a fastText model trained on the parameters specified in the function parameters.  
    """
    ft_model = FastText(
        sg=skipgram,
        size=size,
        window=window, 
        min_count=minimum_count, 
        negative=negative_subsampling, 
        min_n=minimum_n,
        max_n=maximum_n
    )

    ft_model.build_vocab(tokens)

    ft_model.train(
        tokens,
        epochs=epochs, 
        total_examples=ft_model.corpus_count, 
        total_words=ft_model.corpus_total_words
    )

    if save is True: 
        ft_model.save(model_name)
    
    return ft_model


def bm25Vectors(tokens, ft_model, save=True, name="vectors/weighted_doc_vects.p"): 
    """
    PARAMETERS
    ----------
    tokens: list of tokenized words 
    ft_model: fastText model
    save: option to save the bm25 vectors
    name: if save is True, saves the vectors as this 

    RETURNS
    -------
    list of weighted document vectors
    """
    bm25 = BM25Okapi(tokens)
    weighted_doc_vects = []

    for i,doc in tqdm(enumerate(tokens)):
        doc_vector = []
        for word in doc:
            vector = ft_model.wv[word]
            weight = (bm25.idf[word] * ((bm25.k1 + 1.0)*bm25.doc_freqs[i][word])) / (bm25.k1 * (1.0 - bm25.b + bm25.b *(bm25.doc_len[i]/bm25.avgdl))+bm25.doc_freqs[i][word])
            weighted_vector = vector * weight
            doc_vector.append(weighted_vector)
        doc_vector_mean = np.mean(doc_vector,axis=0)
        weighted_doc_vects.append(doc_vector_mean)
    
    if save is True: 
        pickle.dump(weighted_doc_vects, open(name, "wb"))

    return weighted_doc_vects

def HNSWIndex(weighted_doc_vects, save=True, name="index/index.bin"): 
    """
    PARAMETERS
    ----------
    weighted_doc_vects: 
    save: option to save the index
    name: if "save" is True, saves the index as this

    RETURNS
    ------- 
    an hnsw index 
    """
    # create a random matrix to index
    data = np.vstack(weighted_doc_vects)

    # initialize a new index, using a HNSW index on Cosine Similarity - can take a couple of mins
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.addDataPointBatch(data)
    index.createIndex({'post': 2}, print_progress=True)

    if save is True: 
        index.saveIndex(name, save_data=True)
    
    return index

def get_post(query): 
    nlp = spacy.load("en_core_web_sm") # python -m spacy download en_core_web_sm
    tokenized = nlp(query)
    return tokenized

def get_input(document): 
    text = ""
    for token in document: 
        if token.is_alpha and not token.is_stop:  
            text += " " + str(token)
    return text.strip().split()

def time_important(document): 
    # pretty arbitrary list of terms that you can update + change depending on your search engine
    influence = ['influential', 'important', 'pivotal', 'significant', 'leading', 'noteworthy']
    time = ['latest', 'recent', 'newest', 'current']
    text = ""
    important = False
    for token in document: 
        if token not in influence or time: 
            text += " " + str(token)
        else: 
            important = True
    return text.strip(), important

def get_dates(query):
    dates = []
    for ent in filter(lambda e: e.label_=='DATE',query.ents):
        dates.append(ent.text)
    return dates

def get_authors(query):
    authors = []
    for ent in filter(lambda e: e.label_=='PERSON',query.ents):
        authors.append(ent.text)
    return authors

    