# dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_processing import load_and_store_data, engineer_features, create_customer_segments

def main():
    st.title("Customer Behavior Analysis Dashboard")
    
    # Load data
    transactions_df, customers_df = load_and_store_data()
    enhanced_df = engineer_features(transactions_df, customers_df)
    segmented_df = create_customer_segments(enhanced_df)
    
    # Sidebar filters
    st.sidebar.header("Filters")
    selected_segment = st.sidebar.multiselect(
        "Select Customer Segments",
        options=segmented_df['segment'].unique(),
        default=segmented_df['segment'].unique()
    )
    
    # Filter data based on selection
    filtered_df = segmented_df[segmented_df['segment'].isin(selected_segment)]
    
    # Main dashboard components
    col1, col2 = st.columns(2)
    
    with col1:
        # Segment Distribution
        fig_segment = px.pie(filtered_df, 
                           names='segment', 
                           values='total_spend',
                           title='Revenue Distribution by Segment')
        st.plotly_chart(fig_segment)
    
    with col2:
        # Customer Lifetime Value
        fig_clv = px.box(filtered_df, 
                        x='segment', 
                        y='customer_lifetime_value',
                        title='Customer Lifetime Value Distribution')
        st.plotly_chart(fig_clv)
    
    # Seasonal Trends
    st.header("Seasonal Spending Patterns")
    seasonal_cols = [col for col in filtered_df.columns if 'avg_spend_month' in col]
    seasonal_data = filtered_df[seasonal_cols].mean()
    fig_seasonal = px.line(x=range(1, 13), 
                          y=seasonal_data,
                          title='Average Monthly Spending',
                          labels={'x': 'Month', 'y': 'Average Spend'})
    st.plotly_chart(fig_seasonal)
    
    # Customer Metrics Table
    st.header("Customer Segment Metrics")
    segment_metrics = filtered_df.groupby('segment').agg({
        'total_spend': ['mean', 'sum'],
        'customer_lifetime_value': 'mean',
        'total_transactions': 'mean'
    }).round(2)
    st.dataframe(segment_metrics)

if __name__ == "__main__":
    main()
