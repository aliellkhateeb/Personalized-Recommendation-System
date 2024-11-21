import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import hnswlib
import streamlit as st

path = 'All Appliances.csv'
model = SentenceTransformer('all-MiniLM-L6-v2')


def read_data(path):
    # Load data from CSV and combine important features for recommendations
    data = pd.read_csv(path)
    data['combined_features'] = data['name'] + ' ' + \
        data['main_category'] + ' '+data['sub_category']
    return data


def encoding(data, model=model):
    # Encode combined features using a transformer model
    sentences = data['combined_features'].tolist()
    embeddings = model.encode(sentences)
    return embeddings


def recommend_new_product_similarities(new_product_description, hnsw_index, k=10):
    # Create an embedding for the new product description and find top-k similar items
    new_product_embedding = model.encode([new_product_description])[0]
    labels, distances = hnsw_index.knn_query(new_product_embedding, k=k)
    similar_indices = labels.flatten()
    return similar_indices


def display_recommendations(filtered_data, indices):
    # Display the recommendations with enhanced layout
    for idx in indices:
        product = filtered_data.iloc[idx]
        with st.container():
            # Create a row with two columns (image on the left, info on the right)
            col1, col2 = st.columns([1, 4])

            with col1:
                st.image(product['image'], width=150)  # Product image

            with col2:
                st.subheader(product['name'])  # Product name
                st.write(f"Original Price: ${product['actual_price']}")
                st.write(f"Discounted Price: ${product['discount_price']}")


def main():
    st.title("Appliance Recommendation System")

    # Initialize session state for search query
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""

    # Search Bar
    search_query = st.text_input(
        "Search for products", value=st.session_state.search_query)

    # Home button to clear results
    if st.button("Home"):
        st.session_state.search_query = ""  # Clear search query
        st.rerun()  # Re-run the app with cleared state

    # Load data
    data = read_data(path)
    embeddings = np.load('data.npy')  # Load embeddings
    filtered_data = data  # No category filter since working with one category

    # Initialize HNSW index for fast retrieval
    dimension = embeddings.shape[1]
    hnsw_index = hnswlib.Index(space='cosine', dim=dimension)
    hnsw_index.init_index(max_elements=10000, ef_construction=200, M=16)
    hnsw_index.add_items(embeddings)
    hnsw_index.set_ef(50)

    # Display recommendations when a search query is entered
    if search_query:
        st.session_state.search_query = search_query  # Store the query in session state
        indices = recommend_new_product_similarities(
            search_query, hnsw_index, k=10)
        display_recommendations(filtered_data, indices)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Appliance Recommendation System", page_icon="🔌", layout="wide")
    main()
