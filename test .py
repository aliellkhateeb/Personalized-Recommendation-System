import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import hnswlib
import streamlit as st

path = 'D:\Ahmed\GitHub projects\personalised-recommendation-system\All Appliances.csv'
model = SentenceTransformer('all-MiniLM-L6-v2')


def read_data(path):
    data = pd.read_csv(path)
    data['combined_features'] = data['name'] + ' ' + \
        data['main_category'] + ' '+data['sub_category']
    return data


selected_category = 'appliances'
data = read_data(path)
print('read data done')
selected_data = data[data['main_category'] == selected_category]
print('select data done')


sentences = selected_data['combined_features'].to_list()
embeddings = model.encode(sentences)
print("encoding data done")

dim = embeddings.shape[1]  # Dimension of embeddings
print('dim:', dim)

p = hnswlib.Index(space='l2', dim=dim)
p.init_index(max_elements=10000, ef_construction=200, M=16)
p.add_items(embeddings)
p.set_ef(50)  # ef should always be > k


# Query HNSW index for most similar sentence
new_sentence = "mixer"
new_embedding = model.encode([new_sentence])

# Fetch k neighbors
labels, distances = p.knn_query(new_embedding, k=5)
labels = labels.flatten()

print(selected_data.iloc[labels[0]])
