import streamlit as st
import pandas as pd 
import numpy as np 
import plotly.express as px

from wordcloud import WordCloud
import matplotlib.pyplot as plt 

import re 
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')


st.set_page_config(
    page_title='Product & Category Analysis', 
    page_icon='📦', 
    layout='wide', 
    initial_sidebar_state= 'expanded'
)

def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError : 
        nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def extract_keywords(text):
    if not isinstance(text, str):
        return []
    tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
    return [
        word for word in tokens 
        if word not in stop_words and len(word) > 2
    ]


def load_and_clean_data():
    df = pd.read_csv("amazon.csv")

    # clean price
    for col in ['discounted_price', 'actual_price']: 
        df[col] = df[col].str.replace('₹','').str.replace(',','').astype(float)

    # clean percentage
    df['discount_percentage'] = df['discount_percentage'].str.replace('%','').astype(float)

    # clean rating
    df['rating'] = df['rating'].str.replace('|', '3.9').astype(float)

    # clean rating count
    df['rating_count'] = df['rating_count'].str.replace(',', '', regex=False).astype(float)

    median_rating_count = df['rating_count'].median()
    df['rating_count'] = df['rating_count'].fillna(median_rating_count)

    df.dropna(subset=['category'], inplace=True)

    df['main_category'] = df['category'].apply(lambda x: x.split('|')[0])

    return df

def product_analysis_page():
    st.title('📦 Product & Category Deep Dive')
    st.markdown(""" 
    This page analyzes product categories, popular product names and common keywords found in product titles
    """)
    try: 
        data = load_and_clean_data()
    except Exception as e:
        st.error(f"Lỗi: {e}")  
        st.stop()

    st.sidebar.header('Filters')

    all_categories = data['main_category'].unique().tolist()

    selected_categories = st.sidebar.multiselect(
        'Select Category', 
        options=all_categories, 
        default=all_categories
    )

    rating_range = st.sidebar.slider(
        'Rating Range', 
        min_value=float(data['rating'].min()), 
        max_value=float(data['rating'].max()), 
        value=(float(data['rating'].min()), float(data['rating'].max())), 
        step=0.1
    )

    price_range = st.sidebar.slider(
        'Discounted Price', 
        min_value=float(data['discounted_price'].min()), 
        max_value=float(data['discounted_price'].max()), 
        value=(float(data['discounted_price'].min()), float(data['discounted_price'].max())), 
        step=100.0
    )

    filtered_data = data[
        (data['main_category'].isin(selected_categories)) &
        (data['rating'].between(rating_range[0], rating_range[1])) &
        (data['discounted_price'].between(price_range[0], price_range[1]))
    ]  

    if filtered_data.empty: 
        st.warning('No data available')
        st.stop()  

    st.header('Analysis by Product Category')

    category_agg = filtered_data.groupby('main_category').agg(
    average_rating=('rating', 'mean'), 
    average_discount=('discount_percentage', 'mean'),
    total_product=('product_id', 'count')
      ).reset_index()

    st.subheader('Product Count by Main Category')

    col_graph1, col_data_1 = st.columns(2)

    with col_graph1: 
        fig_cat_count = px.bar(
        category_agg.sort_values(by='total_product', ascending=False),
        y='main_category', 
        x='total_product',
        orientation='h', 
        title='Number of Products per Main Category', 
        labels={'total_product':'Number of Products', 'main_category':'Main Category'}, 
        color='total_product',
        color_continuous_scale=px.colors.sequential.Plasma
    )
        st.plotly_chart(fig_cat_count, use_container_width=True)

    with col_data_1:
        st.markdown("### Category Metrics")
        st.dataframe(category_agg.sort_values(by='total_product', ascending=False))


    st.subheader('Most Popular Products')

    popular_products = filtered_data.sort_values(by='rating_count', ascending=False).head(10)

    col_graph2, col_data_2 = st.columns(2)

    with col_graph2: 
        fig_popular = px.bar(
        popular_products,
        x='rating_count', 
        y='product_name',
        orientation='h', 
        title='Top 10 Most Popular Products by Rating Count', 
        labels={'rating_count':'Number of Ratings', 'product_name':'Product Name'}, 
        color='rating_count',
        color_continuous_scale=px.colors.sequential.Plasma
    )
        st.plotly_chart(fig_popular, use_container_width=True)

    with col_data_2:
        st.markdown("### Top 10 Most Popular Products")
        st.dataframe(popular_products[['product_name','main_category','rating','rating_count','discounted_price']])

    st.subheader('Common Keywords in Product Name')

    filtered_data = filtered_data.copy()
    filtered_data['key_words'] = filtered_data['product_name'].apply(extract_keywords)

    all_keywords = [kw for sublist in filtered_data['key_words'] for kw in sublist]

    if all_keywords: 
     text = ' '.join(all_keywords)

     wordcloud = WordCloud(
        width=800, 
        height=700, 
        background_color='white', 
        colormap='viridis', 
        max_words=100
     ).generate(text)

     col_graph3, col_data_3 = st.columns(2)

     with col_graph3: 
        fig, ax = plt.subplots(figsize=(12,6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

     with col_data_3: 
        st.markdown("### Top Product Name Keywords") 
        keyword_counts = pd.Series(all_keywords).value_counts().head(10).reset_index()
        keyword_counts.columns = ['Keywords', 'Frequency']
        st.dataframe(keyword_counts)

    else: 
         st.warning('Could not generate keywords to create a word cloud')
         

if __name__ == '__main__': 
    product_analysis_page()   

