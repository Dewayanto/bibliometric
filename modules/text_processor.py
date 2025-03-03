from wordcloud import WordCloud
import matplotlib.pyplot as plt
import logging

class TextProcessor:
    def __init__(self):
        self.stopwords = set(['and', 'or', 'the', 'in', 'on', 'at', 'to'])
        self.logger = logging.getLogger(__name__)

    def generate_wordcloud(self, keywords, output_path):
        """Generate word cloud visualization from keyword frequencies"""
        if not keywords:
            self.logger.warning("No keywords provided for word cloud generation")
            return

        try:
            # Create word cloud
            wordcloud = WordCloud(
                width=1200,
                height=800,
                background_color='white',
                max_words=100,
                stopwords=self.stopwords
            ).generate_from_frequencies(keywords)

            # Save word cloud
            plt.figure(figsize=(15, 10))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Keyword Cloud', fontsize=16, pad=20)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.logger.error(f"Error generating word cloud: {str(e)}")
            raise

    def process_keywords(self, text):
        """Process and clean keywords"""
        if not isinstance(text, str):
            return []

        # Split and clean keywords
        keywords = [word.strip().lower() for word in text.split(';')]
        # Remove stopwords and short terms
        keywords = [word for word in keywords if word and len(word) > 2 and word not in self.stopwords]
        return keywords