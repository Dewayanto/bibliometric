import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Visualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self._set_style()

    def _set_style(self):
        """Set default visualization style"""
        sns.set_theme()
        sns.set_palette("Set2")

    def plot_publication_trend(self, year_counts, predictions=None):
        """Plot publication trends over time with predictions"""
        plt.figure(figsize=(12, 6))

        # Plot historical data
        ax = year_counts.plot(kind='bar', color='skyblue', label='Historical Data')

        # Plot predictions if available
        if predictions is not None:
            # Add predictions as red bars
            pred_positions = range(len(year_counts), len(year_counts) + len(predictions))
            plt.bar(pred_positions, predictions['Predicted_Publications'], 
                   color='salmon', alpha=0.7, label='AI Predictions')

            # Update x-axis labels
            all_years = list(year_counts.index) + list(predictions['Year'])
            plt.xticks(range(len(all_years)), all_years, rotation=45)

        plt.title('Research Publication Trends and AI Predictions', pad=20)
        plt.xlabel('Publication Year')
        plt.ylabel('Number of Publications')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'publication_trend.png', dpi=300, bbox_inches='tight')
        plt.close()