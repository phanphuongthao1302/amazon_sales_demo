import streamlit as st
import pandas as pd 
import numpy as np 
import plotly.express as px

st.set_page_config(
    page_title='Price & Discount Analysis', 
    page_icon='💰', 
    layout='wide', 
    initial_sidebar_state= 'expanded'
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

def price_analysis_page():
    st.title('💰 Price & Discount Deep Dive')

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

    discount_range = st.sidebar.slider(
        'Discounted Percentage Price', 
        min_value=float(data['discount_percentage'].min()), 
        max_value=float(data['discount_percentage'].max()), 
        value=(float(data['discount_percentage'].min()), float(data['discount_percentage'].max())), 
        step=1.0
    )
    filtered_data = data[
        (data['main_category'].isin(selected_categories)) &
        (data['rating'].between(rating_range[0], rating_range[1])) &
        (data['discounted_price'].between(price_range[0], price_range[1])) &
        (data['discount_percentage'].between(discount_range[0], discount_range[1]))
    ]  

    if filtered_data.empty: 
        st.warning('No data available')
        st.stop()

    
    st.header('Price Correlation Analysis')

    col_graph1, col_data_1 = st.columns(2)

    with col_graph1:
        fig_price_corr = px.scatter(
            filtered_data, 
            x='actual_price', 
            y='discounted_price',
            opacity=0.6, 
            trendline='ols', 
            trendline_color_override='red',
            labels={
                'actual_price': 'Actual Price',
                'discounted_price': 'Discounted Price'
            }, 
        )

        st.plotly_chart(fig_price_corr, use_container_width=True)

    with col_data_1:
        st.markdown("### Descriptive Statistics")
        st.dataframe(filtered_data[['actual_price', 'discounted_price']].describe())


    st.header('Discount % vs. Popularity')
    st.markdown("""Does a higher discount mean more ratings?""")

    col_graph3, col_data_4 = st.columns(2)

    with col_graph3:
        fig_discount_ratingcnt = px.scatter(
            filtered_data, 
            x='discount_percentage', 
            y='rating_count',
            opacity=0.6, 
            labels={
                'discount_percentage': 'Discount Percentage',
                'rating_count': 'Number of Rating'
            }
        )

        st.plotly_chart(fig_discount_ratingcnt, use_container_width=True)

    with col_data_4:
        st.markdown("### Descriptive Statistics")
        st.dataframe(filtered_data[['discount_percentage', 'rating_count']].describe())

    discount_corr = filtered_data['discount_percentage'].corr(filtered_data['rating_count'])    
    st.markdown(f"""
     This plot reveals a very weak correlation (coefficient:`{discount_corr:.2f}`) between discount percentage and number of ratings.
     This suggests that simply offering a higher discount does not necessarily lead to significant increase in product popularity or customer reviews
     """)

    st.header('Price vs. Customer Rating')
    st.markdown("""Does a higher price tag mean a better rating? Let's investigate!""")

    col_graph5, col_data_6 = st.columns(2)

    with col_graph5:
        fig_price_rating = px.scatter(
            filtered_data, 
            x='actual_price', 
            y='rating',
            opacity=0.6, 
            labels={
                'actual_price': 'Actual Price',
                'rating': 'Rating (out of 5)'
            }
        )

        st.plotly_chart(fig_price_rating, use_container_width=True)

    with col_data_6:
        st.markdown("### Descriptive Statistics")
        st.dataframe(filtered_data[['actual_price', 'rating']].describe())

    price_rating_corr = filtered_data['actual_price'].corr(filtered_data['rating'])    
    st.markdown(f"""
     The correlation between price and product rating is very weak (coefficient:`{price_rating_corr:.2f}`.
     This indicates that customers rate products more on their quality and experience rather than just their price point)
     """)   
if __name__ == '__main__': 
    price_analysis_page()
