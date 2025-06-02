# src/advanced_visualization.py
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class AdvancedVisualization:
    @staticmethod
    def create_customer_journey_map(df: pd.DataFrame) -> go.Figure:
        """
        Create an interactive customer journey map.
        """
        # Prepare data
        journey_data = df.groupby('month')['amount'].agg(['mean', 'count']).reset_index()
        
        # Create figure
        fig = go.Figure()
        
        # Add traces
        fig.add_trace(go.Scatter(
            x=journey_data['month'],
            y=journey_data['mean'],
            mode='lines+markers',
            name='Average Spend'
        ))
        
        fig.add_trace(go.Bar(
            x=journey_data['month'],
            y=journey_data['count'],
            name='Transaction Count',
            yaxis='y2'
        ))
        
        # Update layout
        fig.update_layout(
            title='Customer Journey Map',
            xaxis=dict(title='Month'),
            yaxis=dict(title='Average Spend'),
            yaxis2=dict(title='Transaction Count', overlaying='y', side='right'),
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def create_cohort_analysis(df: pd.DataFrame) -> go.Figure:
        """
        Create cohort analysis visualization.
        """
        # Prepare cohort data
        df['cohort'] = df['first_purchase_date'].dt.to_period('M')
        df['month_number'] = ((df['transaction_date'].dt.to_period('M') - 
                             df['first_purchase_date'].dt.to_period('M')).astype(int))
        
        cohort_data = df.groupby(['cohort', 'month_number'])['customer_id'].nunique()
        cohort_data = cohort_data.unstack()
        
        # Calculate retention rates
        retention_rates = cohort_data.divide(cohort_data[0], axis=0) * 100
        
        # Create heatmap
        fig = px.imshow(
            retention_rates,
            labels=dict(x='Month Number', y='Cohort', color='Retention Rate'),
            title='Customer Cohort Analysis'
        )
        
        return fig
    
    @staticmethod
    def create_customer_segment_profile(df: pd.DataFrame) -> go.Figure:
        """
        Create detailed customer segment profiles.
        """
        # Calculate segment profiles
        profiles = df.groupby('segment').agg({
            'total_spend': 'mean',
            'customer_lifetime_value': 'mean',
            'total_transactions': 'mean',
            'avg_transaction_value': 'mean'
        }).round(2)
        
        # Create radar chart
        fig = go.Figure()
        
        for segment in profiles.index:
            fig.add_trace(go.Scatterpolar(
                r=profiles.loc[segment],
                theta=profiles.columns,
                name=f'Segment {segment}'
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title='Customer Segment Profiles'
        )
        
        return fig
