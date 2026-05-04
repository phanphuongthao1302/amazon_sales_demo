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
    page_title='Customer Review Insights', 
    page_icon='🗣️', 
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

def review_analysis_page():
    st.title('🗣️ Customer Review Insights')
    st.markdown(""" 
    Explore customer sentiment throught rating and review content. What do high rating and popularity tells us? What are customers talking about? 
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

    st.header('How Do Ratings Correlate With Popularity?')
    st.markdown(f"""
    This chart investigates whether a higher rating corresponds to a higher number of reviews (`rating_count`).
    """)

    col_graph1, col_data_1 = st.columns(2)

    with col_graph1: 
        fig_rating_pop = px.scatter(
        filtered_data,
        x='rating', 
        y='rating_count',
        orientation='h', 
        title='Rating vs. Popularity', 
        labels={'rating':'Average Rating', 'rating_count':'Number of Ratings'}, 
        color='rating',
        color_continuous_scale=px.colors.sequential.Viridis, 
        hover_data = ['product_name']
    )
        st.plotly_chart(fig_rating_pop, use_container_width=True)
    with col_data_1: 
        st.markdown('##### Discriptive Satistics')  
        st.dataframe(filtered_data[['rating','rating_count']].describe())  
    rating_pop_corr = filtered_data['rating'].corr(filtered_data['rating_count'])  
    st.markdown(f"""
    This scatter plot shows a weak positive correlation (coefficient:`{rating_pop_corr:.2f}`) between product ratings and the number of reviews.
    While popular products tend to maintain good ratings (often above 4.0), a high rating alone doesn't guarantee widespread popularity.
    """)   

    st.subheader('Most Frequent Work in Customer Reviews')
    st.markdown(""" This word cloud visualizes the most common terms found in the 'review_content', highlights key themes and customer feedbacks""" )

    review_text = filtered_data['review_content'].dropna()
    all_keywords = []

    if not review_text.empty:
        filtered_data['keywords'] = review_text.apply(extract_keywords)
        all_keywords = [kw for sublist in filtered_data['keywords'] for kw in sublist]

    if all_keywords: 
        text = ' '.join(all_keywords)
        wordcloud = WordCloud( width=800, height=700, background_color='white', colormap='viridis', max_words=100).generate(text)

        col_graph3, col_data_3 = st.columns(2)

        with col_graph3: 
             fig, ax = plt.subplots(figsize=(12,6))
             ax.imshow(wordcloud, interpolation='bilinear')
             ax.axis('off')
             st.pyplot(fig)
             
        with col_data_3: 
            st.markdown("### Top Review Keywords") 
            keyword_counts = pd.Series(all_keywords).value_counts().head(10).reset_index()
            keyword_counts.columns = ['Keywords', 'Frequency']
            st.dataframe(keyword_counts)
        st.markdown("""
        This word cloud visually presents the most common words found in product names.
        Larger words appear frequently, indicating key products features or types that resonate with customers
        """)
    else: 
         st.warning('Could not generate keywords to create a word cloud') 


if __name__ == '__main__': 
    review_analysis_page()
