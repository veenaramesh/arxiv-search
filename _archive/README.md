# arXiv search engine
This code is now outdated as of Feb 2024. 

## Using the search engine
This search engine allows you to search efficiently through all of the arXiv articles. 

## Running the application locally 

First, create a virtual environment with conda or venv inside a temp folder, then activate it. 

```
conda create -n your-env-name 

source activate your-env-name
```

Clone the git repo, then install the requirements with pip. If you are using conda environment, you will need to install pip separately. 

```
pip install -r requirements.txt
```

Then, run the setup.py file. This will take up to an hour. This sets up the fastText models and indices that are necessary for the search engine. If you want to make changes to the parameters in the fastText models, you can do so here. 

Finally, run the app. 

```
python setup.py

python app.py
```

## Behind the Scenes
### Motivation

Although modern information retrieval (IR) systems are designed for indexing and searching unstructured data, including text and image data, the application of such systems oriented around retrieving structured data is still an important focus of research. Traditional information retrieval systems diverge from relational databases and their associated querying methods in terms of the retrieval models used, the underlying data structures, and the features of the query language. The integration of natural language querying already familiar to unstructured data searching procedures and traditional SQL-like search schema for querying structured data presents several challenges for researchers. Effective query parsing and translation serve as the cornerstone for such applications.

This presents an application of a search engine for structured relational data in the context of **querying arXiv**, an open-access archive for over 1.8 million scholarly articles in a variety of fields. The framework presented in this paper aims to extend the existing functionality of the arXiv search system, with the goal of increasing accessibility for novice users and the quality of the results list for experienced users.

The dataset can be found here: https://www.kaggle.com/Cornell-University/arxiv

### Dataset Description

The ArXiv have millions of publications which provide high quality data for the project. The scientific publication are categorized with International Catalogue of Scientific Literature. Each publication will have it own category before submit to the database. The category of each publication provide an good way to build the structured database for the system.

ArXiv API provide a mirror of the original ArXiv data. The original dataset is rather large(more than 1.1 TB). To reduce the system requirement of the demo system, it will only use the metadata for the system. The dataset are in Json format contain an entry for each paper, containing: 

    - id: the ArXiv ID which can be used for access the original paper
    - submitter: The submitter of the paper
    - authors: Authors of the paper 
    - title: The title of the paper 
    - comments: Additional info, such as number of pages and figures
    - journal-ref: Information about the journal the paper was published in
    - abstract: The abstract of the paper
    - categories: Categories / tags in the ArXiv system
    
### Query Parsing and Indexing 

There are two aspects of the search engine that need to be addressed: (1) the input query and (2) the structured relational data. The input query needs to be parsed properly in order to be properly understood by the search engine, and the structural relational data needs to be indexed to be quickly search-able. These two aspects are combined to produce proper search results. 

The spaCy library offers an extensive amount of advanced NLP functions in Python. The query was first tokenized, i.e. segments into words, punctuation, etc. The rules are specific to each language, and because all arXiv titles and abstracts are in English, the queries were also tokenized in English. The query is now transformed into a Doc object that contains each token. Through this library, the query can be parsed out to understand: (1) the part of speech, (2) entity recognition, and (3) date recognition. 

There were four main steps to creating the index: (1) tokenizing, (2) creating word embeddings, (3) vectorizing the documents, and (4) actually producing the index. The arXiv search engine should be able to return relevant results to users even if specific words were not within those results, be able to scale to larger datasets, and handle spelling mistakes and out-of-vocabulary words.

Consider the features in the arXiv data. There are only a few of those categories that were legitimately important in searching: (1) authors, (2) titles, (3) abstracts, and (4) categories. The first step is splitting the documents into tokens. This process is similar to the query parsing step. Each of columns of the features mentioned above were tokenized through the spaCy library. In order to perform efficiently, much of the advanced properties offered in the spaCy library, like part-of-speech tagging, were disabled. Here, the spaCy library was used to simply tokenize and clean the document.
Using the Gensim library and the fastText model, it can create a word vector space model that not only is computational efficient but is able to capture the relationships between words quite well. FastText leverages the principle that states that the morphological structure of a word contains important information about the meaning of the word. Word2Vec, in contrast, does not take these morphological structures into account. This is quite important to morphologically rich languages, like German and English, where a word has multiple forms. FastText also has the ability to obtain vectors for out-of-vocabulary (OOV) words. This is especially important in the arXiv search engine, considering many words in the STEM vocabulary are not considered in modern dictionaries.

Through fastText, it have obtained word vectors. In order to obtain a combination of word vectors for each document (or entry in a column), it leveraged the BM25 algorithm. While usual attempts generally average the word vectors for each document, it has been shown that combining fastText word vectors with the BM25 algorithm performs significantly better and provides higher quality search results. BM25 implements two refinements on the original TF-IDF algorithm: (1) term frequency saturation and (2) document length. The first is fairly intuitive; BM25 ensures that there are diminishing returns for the number of terms matched in documents. When looking for a specific term that is common in documents, there should be a point where the number of occurrences is less useful. The second is also fairly intuitive. If a shorter article contains the same amount of terms that match as a longer article, then it is more likely that the shorter article is more relevant. BM25 takes this document length into account when matching. This introduces two hyper-parameters that impact the ranking functions: 'k' to tune the impact term saturation and 'b' to tune the impact of document length. 

The BM25 algorithm has outputted a list of vectors for each document in the dataset. In order to search efficiently through the dataset. The vectors for each document are high dimensional, and searching through each of these dimensions over a dataset of more than 1.8 million examples is computationally expensive. Therefore, it used the NMSLIB library to create a HNSW index. Using a cosine similarity function, the Hierarchical Navigable Small World (HNSW) index allows us to perform a K-nearest-neighbors query. The query is vectorized through the fastText model and averaged to search through the index. 

As previously mentioned, there were several different columns or features. These columns were: (1) title, (2) abstract, (3) authors, and (3) categories. The date was parsed out to create two different columns, month and year. In order to properly search through these values, four different models were created to properly search through the title, abstract, authors, and categories. The query was parsed, as discussed in the section Query Parsing, and different parts of the text were submitted to different indices based. For example, for the query text "what's the recent research on neutron star richard jenkins physics," the query parser recognized "richard jenkins" as a name and submitted that text through to the author index. The query parser also recognized "physics" as a category and submitted that text through to the category index. The query parser submits the entire query, with the author and categories removed, to both the title and abstract index. 

The search results found for each of the indices are then aggregated. The top 10 results, based on the distances calculated, are outputted. 

Dates are incorporated in the search engine in a slightly different way. Unless the user specifies a date or states that dates are important in some way (i.e. using keywords like 'recent' or 'latest'), the search results are not organized by dates. Specific dates allow the search engine to simply splice the data-frame and output results in that manner. 
