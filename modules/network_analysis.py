import networkx as nx
from itertools import combinations
import logging

class NetworkAnalyzer:
    def __init__(self, data):
        self.data = data
        self.logger = logging.getLogger(__name__)

    def create_author_network(self):
        """Create simple author collaboration network"""
        G = nx.Graph()

        for _, row in self.data.iterrows():
            if isinstance(row['Authors'], str):
                # Split and clean author names
                authors = [a.strip() for a in row['Authors'].split(';') if a.strip()]

                # Add edges between all author pairs
                for author1, author2 in combinations(authors, 2):
                    if G.has_edge(author1, author2):
                        G[author1][author2]['weight'] += 1
                    else:
                        G.add_edge(author1, author2, weight=1)

        return G