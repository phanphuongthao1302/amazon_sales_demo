import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(
    page_title='Product Recommender',
    layout='wide',
    initial_sidebar_state='expanded'
)

@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("amazon.csv")

    for col in ['discounted_price', 'actual_price']:
        df[col] = df[col].str.replace('₹', '').str.replace(',', '').astype(float)

    df['discount_percentage'] = df['discount_percentage'].str.replace('%', '').astype(float)
    df['rating'] = df['rating'].str.replace('|', '3.9').astype(float)
    df['rating_count'] = df['rating_count'].str.replace(',', '', regex=False).astype(float)

    df['rating_count'] = df['rating_count'].fillna(df['rating_count'].median())
    df.dropna(subset=['category'], inplace=True)

    df['main_category'] = df['category'].apply(lambda x: x.split('|')[0])

    # Tạo full_text từ các cột text
    df['full_text'] = (
        df['product_name'].fillna('') + ' ' +
        df['about_product'].fillna('') + ' ' +
        df['review_content'].fillna('')
        ).str.lower()


    def extract_name(text, pattern):
        match = re.search(pattern, text, re.IGNORECASE)
        return float(match.group(1)) if match else None

    def extract_features(row):
        text = str(row['full_text']).lower()
        features = {}

        features['watts'] = extract_name(text, r'(\d+)(\.\d+)?\s*w')
        features['typec'] = int(('type-c' in text) or ('type c' in text) or ('usb-c' in text) or ('usb c' in text))
        features['fast_charge'] = int(any(kw in text for kw in [
            'fast charging', 'fast charge', 'quick charge', 'qc 3.0', 'qc3.0', 'power delivery', 'pd', 'rapid charge'
        ]))
        features['braided'] = int('braided' in text)

        colors = ['black', 'white', 'red', 'blue', 'green']
        for c in colors:
            features[f'color_{c}'] = int(c in text)

        return features

    feature_df = df.apply(extract_features, axis=1, result_type='expand')

    numeric_cols = [
        'watts', 'typec', 'fast_charge', 'braided',
        'color_black', 'color_white', 'color_red', 'color_blue', 'color_green'
    ]

    for col in numeric_cols:
        if col not in feature_df.columns:
            feature_df[col] = 0

    df = pd.concat([df, feature_df], axis=1)
    df[numeric_cols] = df[numeric_cols].fillna(0).astype(float)

    return df, numeric_cols

@st.cache_data
def build_similarity_matrices(df, numeric_cols):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    text_matrix = tfidf.fit_transform(df['full_text'].fillna(''))
    text_sim = cosine_similarity(text_matrix)

    scaler = MinMaxScaler()
    numeric_scaled = scaler.fit_transform(df[numeric_cols])
    numeric_sim = cosine_similarity(numeric_scaled)

    return text_sim, numeric_sim

def get_similarity_score(df, idx1, idx2, text_sim, num_sim):
    text_score = text_sim[idx1, idx2]
    num_score = num_sim[idx1, idx2]

    cat_boost = 1.2 if df.loc[idx1, "main_category"] == df.loc[idx2, "main_category"] else 1.0

    return (0.6 * text_score + 0.4 * num_score) * cat_boost

def recommend(df, product_name, text_sim, num_sim, top_n=5):
    if product_name not in df['product_name'].values:
        return None

    idx = df.index[df['product_name'] == product_name][0]

    scores = [
        (i, get_similarity_score(df, idx, i, text_sim, num_sim))
        for i in range(len(df)) if i != idx
    ]

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    top_indices = [i[0] for i in scores[:top_n]]

    return df.loc[top_indices, [
        "product_name", "main_category", "discounted_price", "rating"
    ]].reset_index(drop=True)

def recommender_page():
    st.title("🎯Product Recommendation Engine")   
    st.markdown("""
    Select a product from the dropdown below to see a list of the top 5 most similar products
    The recommendation is based on a combination of product description (text similarity) and key features (feature similarity)""")

    try:
        data, numeric_features = load_and_clean_data()
        text_sim_matrix, numeric_sim_matrix = build_similarity_matrices(data, numeric_features)
    except Exception as e:
        st.error(f"Lỗi: {e}")  
        st.stop()

    product_list = data['product_name'].unique()

    select_product = st.selectbox(
        "Choose a product to Get Recommendation For:",
        product_list,
        index=0
    )

    if st.button('Find similarity products'):
        with st.spinner('Finding recommendation...'):
            recommendations = recommend(data, select_product, text_sim_matrix, numeric_sim_matrix)

            st.subheader(f'Top 5 Recommendations for {select_product} are:')

            if recommendations is not None and not recommendations.empty:
                st.table(recommendations[["product_name","main_category","discounted_price", "rating"]])
                st.markdown("""
                The table above presents the top 5 products most similar to your selected item. 
                These recomendations are generated by analyzing both textual descriptions and key products features, aiming to suggest items that align with your interest.""")
            else:
                st.warning('Could not find any recommendations for this product')

# RUN
if __name__ == '__main__': 
    recommender_page()

