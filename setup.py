""" 
RUN THIS SCRIPT BEFORE YOU RUN app.py 

This sets up the fastText models + indices that are necessary for the actual search engine
""" 

import pandas as pd
import utils

# read in the dataset
df = pd.read_csv("data/arxiv_smaller.csv")
# specify the columns that you want to search
df_columns = ["title", "author", "categories"]

# the model, the vectors, and the indices will be saved in their respective folders  
# (change save=False if you do not want them saved)
for column in df_columns: 
    tokens = utils.tokenize(column)
    model = utils.creatingModel(tokens, skipgram=1, size=100, window=10, minimum_count=5, negative_subsampling=15, minimum_n=2, maximum_n=5, epochs=6, model_name="models/_fasttext_"+column+".model") 
    vectors = utils.bm25Vectors(tokens, model, name="vectors/weighted_doc_vects_"+column+".p")
    index = utils.HNSWIndex(vectors, name="index/index_"+column+".bin")