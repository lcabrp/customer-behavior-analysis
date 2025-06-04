# src/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Visualizer:
    def __init__(self):
        # Set default style
        """
        Initialize the Visualizer class with default styles and figure sizes.

        Sets the default plotting style to 'fast' using matplotlib, applies the
        'whitegrid' theme with increased font scale using seaborn, and sets the 
        color palette to 'husl'. Also defines default figure sizes for single 
        plots and subplots.
        """
        sns.set_theme(style='whitegrid', font_scale=1.5)
      
       # ['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 
       # 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'petroff10', 'seaborn-v0_8', 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 
       # 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 
       # 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper', 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 
       # 'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid', 'tableau-colorblind10']
         
        # Set default style
        #plt.style.use('fast')  # Use 'fast' for better performance in matplotlib
        # Use a modern color palette
        self.color_palette = sns.color_palette("husl")
        sns.set_palette(self.color_palette)

        # Default figure sizes
        self.default_figsize = (12, 6)
        self.default_subplot_figsize = (15, 10)

    def plot_distribution(self,
                          data: pd.Series,
                          title: str,
                          kde: bool = True,
                          figsize: Optional[Tuple[int, int]] = None) -> None:
        """
        Plot distribution of a variable.

        Parameters:
            data (pd.Series): Data to plot
            title (str): Plot title
            kde (bool): Whether to include KDE plot
            figsize (tuple): Figure size
        """
        plt.figure(figsize=figsize or self.default_figsize)
        sns.histplot(data=data, kde=kde)
        plt.title(title)
        plt.xlabel(data.name)
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()

    def plot_time_series(self,
                         data: pd.DataFrame,
                         date_column: str,
                         value_column: str,
                         title: str,
                         figsize: Optional[Tuple[int, int]] = None) -> None:
        """
        Plot time series data.

        Parameters:
            data (pd.DataFrame): DataFrame containing time series data
            date_column (str): Name of date column
            value_column (str): Name of value column
            title (str): Plot title
            figsize (tuple): Figure size
        """
        plt.figure(figsize=figsize or self.default_figsize)

        # Group by date and calculate mean
        time_series = data.groupby(date_column)[value_column].mean()

        plt.plot(time_series.index, time_series.values, marker='o')
        plt.title(title)
        plt.xlabel(date_column)
        plt.ylabel(value_column)
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_segment_analysis(self,
                              data: pd.DataFrame,
                              segment_column: str,
                              metric_columns: List[str],
                              title: str) -> None:
        """
        Create segment analysis plots.

        Parameters:
            data (pd.DataFrame): Segmented customer data
            segment_column (str): Name of segment column
            metric_columns (List[str]): List of metrics to analyze
            title (str): Plot title
        """
        n_metrics = len(metric_columns)
        fig, axes = plt.subplots(1, n_metrics,
                                figsize=(5 * n_metrics, 6))

        if n_metrics == 1:
            axes = [axes]

        for i, metric in enumerate(metric_columns):
            sns.boxplot(data=data, x=segment_column, y=metric, ax=axes[i])
            axes[i].set_title(f'{metric} by Segment')
            axes[i].tick_params(axis='x', rotation=45)

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def plot_correlation_matrix(self,
                               data: pd.DataFrame,
                               figsize: Optional[Tuple[int, int]] = None) -> None:
        """
        Plot correlation matrix heatmap.

        Parameters:
            data (pd.DataFrame): Numerical data for correlation analysis
            figsize (tuple): Figure size
        """
        plt.figure(figsize=figsize or self.default_figsize)

        # Calculate correlations
        corr = data.corr()

        # Create heatmap
        sns.heatmap(corr,
                   annot=True,
                   cmap='coolwarm',
                   center=0,
                   fmt='.2f')

        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()

    def plot_customer_segments_3d(self,
                                 data: pd.DataFrame,
                                 x_col: str,
                                 y_col: str,
                                 z_col: str,
                                 segment_col: str) -> None:
        """
        Create 3D scatter plot of customer segments.

        Parameters:
            data (pd.DataFrame): Segmented customer data
            x_col (str): Column name for x-axis
            y_col (str): Column name for y-axis
            z_col (str): Column name for z-axis
            segment_col (str): Column name for segments
        """
        fig = px.scatter_3d(data,
                           x=x_col,
                           y=y_col,
                           z=z_col,
                           color=segment_col,
                           title='Customer Segments 3D Visualization')

        fig.update_layout(scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col
        ))

        fig.show()

    def plot_seasonal_patterns(self,
                              data: pd.DataFrame,
                              date_column: str,
                              value_column: str) -> None:
        """
        Plot seasonal patterns in the data.

        Parameters:
            data (pd.DataFrame): Time series data
            date_column (str): Name of date column
            value_column (str): Name of value column
        """
        # Convert date column to datetime if needed
        data[date_column] = pd.to_datetime(data[date_column])

        # Extract time components
        data['year'] = data[date_column].dt.year
        data['month'] = data[date_column].dt.month

        # Create seasonal plot
        plt.figure(figsize=self.default_figsize)

        for year in data['year'].unique():
            year_data = data[data['year'] == year]
            plt.plot(year_data['month'],
                    year_data.groupby('month')[value_column].mean(),
                    marker='o',
                    label=str(year))

        plt.title('Seasonal Patterns')
        plt.xlabel('Month')
        plt.ylabel(value_column)
        plt.legend(title='Year')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_customer_lifecycle(self,
                               data: pd.DataFrame,
                               customer_id_col: str,
                               date_col: str,
                               value_col: str) -> None:
        """
        Plot customer lifecycle analysis.

        Parameters:
            data (pd.DataFrame): Customer transaction data
            customer_id_col (str): Customer ID column name
            date_col (str): Transaction date column name
            value_col (str): Transaction value column name
        """
        # Calculate customer metrics
        customer_data = data.groupby(customer_id_col).agg({
            date_col: ['min', 'max'],
            value_col: ['count', 'sum', 'mean']
        }).reset_index()

        customer_data.columns = [
            customer_id_col, 'first_purchase', 'last_purchase',
            'transaction_count', 'total_spend', 'avg_transaction'
        ]

        # Calculate lifetime days
        customer_data['lifetime_days'] = (
                customer_data['last_purchase'] - customer_data['first_purchase']
        ).dt.days

        # Create subplot figure
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('Customer Lifetime Distribution',
                                          'Transaction Count vs Total Spend',
                                          'Average Transaction Value Distribution',
                                          'Customer Activity Timeline'))

        # Add traces
        fig.add_trace(
            go.Histogram(x=customer_data['lifetime_days'],
                         name='Lifetime Days'),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=customer_data['transaction_count'],
                      y=customer_data['total_spend'],
                      mode='markers',
                      name='Customer Transactions'),
            row=1, col=2
        )

        fig.add_trace(
            go.Box(y=customer_data['avg_transaction'],
                   name='Avg Transaction'),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=customer_data['first_purchase'],
                      y=customer_data['transaction_count'],
                      mode='markers',
                      name='Customer Acquisition'),
            row=2, col=2
        )

        fig.update_layout(height=800, title_text="Customer Lifecycle Analysis")
        fig.show()

    def save_all_plots(self,
                       data: pd.DataFrame,
                       output_dir: str) -> None:
        """
        Save all visualization plots to specified directory.

        Parameters:
            data (pd.DataFrame): Analysis data
            output_dir (str): Output directory path
        """
        import os

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save various plots
        plt.figure(figsize=self.default_figsize)

        # Distribution plots
        for col in data.select_dtypes(include=[np.number]).columns:
            self.plot_distribution(data[col], f'{col} Distribution')
            plt.savefig(os.path.join(output_dir, f'{col}_distribution.png'))
            plt.close()

        # Correlation matrix
        self.plot_correlation_matrix(data.select_dtypes(include=[np.number]))
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
        plt.close()

        # Save other plots as needed
        print(f"All plots saved to {output_dir}")
