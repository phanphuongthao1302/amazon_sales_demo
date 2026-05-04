import streamlit as st
import pandas as pd 
import numpy as np 
import plotly.express as px

st.set_page_config(
    page_title='Amazon Sale Analysis', 
    page_icon='🛒', 
    layout='wide'
)

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


def main(): 
    st.title('🛒 Amazon Sales Analysis')

    try: 
        data = load_and_clean_data()
    except Exception as e:
        st.error(f"Lỗi: {e}")   # 👈 QUAN TRỌNG để debug
        st.stop()

    # Sidebar
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

    # KPI
    st.header('Dataset Overview')

    col1, col2, col3, col4 = st.columns(4)

    total_products = filtered_data.shape[0]
    avg_rating = round(filtered_data['rating'].mean(), 2)
    avg_discount = round(filtered_data['discount_percentage'].mean(), 2)
    total_categories = filtered_data['main_category'].nunique()

    col1.metric('Total Products', f'{total_products:,}')
    col2.metric('Average Rating', f'{avg_rating} ⭐')
    col3.metric('Average Discount', f'{avg_discount}%')
    col4.metric('Categories', total_categories)
    
    st.header('Initial Visual Insights')
    st.subheader('Distribution of Products Rating')
    fig_rating = px.histogram(filtered_data, x = 'rating', nbins = 20, title = 'Product Distribution Rating', labels = {'rating':'Rating'}, color_discrete_sequence = ['blue'])
    fig_rating.update_layout(bargap=0.1)
    st.plotly_chart(fig_rating, config={"responsive": True})

    st.subheader('Distribution of Product Discounts')
    fig_rating = px.histogram(filtered_data, x = 'discounted_price', nbins = 20, title = 'Product Distribution Discounts', labels = {'discounted_price':'Discount(%)'}, color_discrete_sequence = ['green'])
    fig_rating.update_layout(bargap=0.1)
    st.plotly_chart(fig_rating, config={"responsive": True})

    st.subheader('top 10 Most Frequent Product Catagories')
    category_counts = filtered_data['main_category'].value_counts().nlargest(10).reset_index()
    category_counts.columns = ['category', 'count']
    fig_cat = px.bar(category_counts, y='category', x='count', orientation='h', title = 'top 10 Product Categories', color = 'count', color_continuous_scale = px.colors.sequential.Viridis)
    fig_cat.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_cat, config={"responsive": True})

    st.markdown("""
     The bar chart highlights the top 10 most frequent product categories in the dataset.
     Understand this dominant categories can inform inventory manegement and marketing strategies
     """)
if __name__ == '__main__': 
    main()
