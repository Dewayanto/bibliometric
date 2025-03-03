import pandas as pd
from .trend_predictor import TrendPredictor
import logging

class BibliometricAnalyzer:
    def __init__(self, data):
        self.data = data
        self.trend_predictor = TrendPredictor()
        self.logger = logging.getLogger(__name__)

    def get_publication_stats(self):
        """Calculate basic publication statistics"""
        stats = {
            'total_publications': len(self.data),
            'year_range': f"{self.data['Year'].min()}-{self.data['Year'].max()}"
        }
        return stats

    def get_publication_trends(self):
        """Analyze publication trends by year"""
        return self.data['Year'].value_counts().sort_index()

    def predict_future_trends(self, future_years=2):
        """Predict future publication trends"""
        try:
            self.logger.info(f"Starting future trend prediction for {future_years} years")
            current_trends = self.get_publication_trends()
            self.logger.info(f"Current publication trends:\n{current_trends}")

            predictions = self.trend_predictor.predict_publication_trends(current_trends, future_years)
            self.logger.info(f"Generated predictions:\n{predictions}")
            return predictions
        except Exception as e:
            self.logger.error(f"Error in predict_future_trends: {str(e)}")
            raise

    def export_results(self, output_dir):
        """Export analysis results"""
        try:
            self.logger.info("Starting to export results...")

            # Export basic statistics
            stats = self.get_publication_stats()
            pd.DataFrame([stats]).to_csv(output_dir / 'statistics.csv', index=False)
            self.logger.info("Exported statistics")

            # Export publication trends
            trends = self.get_publication_trends()
            trends.to_csv(output_dir / 'publication_trends.csv')
            self.logger.info("Exported publication trends")

            # Export predictions
            predictions = self.predict_future_trends()
            predictions.to_csv(output_dir / 'trend_predictions.csv', index=False)
            self.logger.info(f"Exported predictions:\n{predictions}")

        except Exception as e:
            self.logger.error(f"Error exporting results: {str(e)}")
            raise