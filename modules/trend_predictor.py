import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import logging

class TrendPredictor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def predict_publication_trends(self, year_counts, future_years=2):
        """Predict future publication counts using polynomial regression"""
        try:
            self.logger.info(f"Starting trend prediction for {future_years} future years")
            self.logger.info(f"Historical data: {year_counts.to_dict()}")

            # Prepare data
            years = np.array(list(year_counts.index)).reshape(-1, 1)
            counts = np.array(list(year_counts.values))

            self.logger.info(f"Training data - Years: {years.flatten()}, Counts: {counts}")

            # Fit polynomial regression
            poly = PolynomialFeatures(degree=1)  # Linear trend to avoid overfitting
            X_poly = poly.fit_transform(years)
            model = LinearRegression()
            model.fit(X_poly, counts)

            # Generate future years
            last_year = years[-1][0]
            future_years_range = np.array(range(last_year + 1, last_year + future_years + 1))
            future_X = poly.transform(future_years_range.reshape(-1, 1))

            self.logger.info(f"Predicting for years: {future_years_range}")

            # Make predictions
            predictions = model.predict(future_X)
            self.logger.info(f"Initial predictions: {predictions}")

            # Calculate trend based on last 3 years
            recent_trend = np.mean(np.diff(counts[-3:])) if len(counts) >= 3 else 0
            self.logger.info(f"Recent trend: {recent_trend}")

            # Adjust predictions using recent trend
            for i in range(len(predictions)):
                # Use max between model prediction and trend-based prediction
                trend_pred = max(counts[-1] + recent_trend * (i + 1), 1)
                predictions[i] = max(predictions[i], trend_pred)

            self.logger.info(f"Final adjusted predictions: {predictions}")

            # Create prediction DataFrame with rounded values
            pred_df = pd.DataFrame({
                'Year': future_years_range,
                'Predicted_Publications': predictions.round().astype(int)
            })

            self.logger.info(f"Final prediction DataFrame:\n{pred_df}")
            return pred_df

        except Exception as e:
            self.logger.error(f"Error predicting publication trends: {str(e)}")
            raise