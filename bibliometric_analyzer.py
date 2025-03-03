import argparse
import sys
import logging
from pathlib import Path
from modules.data_loader import DataLoader
from modules.analyzer import BibliometricAnalyzer
from modules.visualizer import Visualizer

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )
    return logging.getLogger(__name__)

def main():
    logger = setup_logging()

    parser = argparse.ArgumentParser(description='Simple Bibliometric Analysis Tool')
    parser.add_argument('input_file', help='Path to input CSV file')
    parser.add_argument('--output', '-o', help='Output directory for results', default='output')
    parser.add_argument('--future-years', '-f', type=int, default=2,
                       help='Number of years to predict (default: 2)')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load and process data
        logger.info("Loading data from %s...", args.input_file)
        loader = DataLoader(args.input_file)
        df = loader.load_data()
        logger.info("Successfully loaded %d records", len(df))

        # Basic statistics
        analyzer = BibliometricAnalyzer(df)
        stats = analyzer.get_publication_stats()
        logger.info("\nBasic Statistics:")
        logger.info(f"Total Publications: {stats['total_publications']}")
        logger.info(f"Year Range: {stats['year_range']}")

        # Generate visualization with predictions
        logger.info("Generating visualization with trend predictions...")
        visualizer = Visualizer(output_dir)
        year_counts = analyzer.get_publication_trends()
        predictions = analyzer.predict_future_trends(args.future_years)
        visualizer.plot_publication_trend(year_counts, predictions)
        logger.info("Created publication trend visualization with predictions")

        # Export results
        logger.info("Exporting results...")
        analyzer.export_results(output_dir)

        # Display predictions
        logger.info("\nPublication Predictions:")
        for _, row in predictions.iterrows():
            logger.info(f"Year {row['Year']}: {row['Predicted_Publications']} publications")

        logger.info(f"Analysis complete! Results saved to {output_dir}")

    except Exception as e:
        logger.error("Error during analysis: %s", str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()