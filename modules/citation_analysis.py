import networkx as nx
import numpy as np
import pandas as pd
from collections import defaultdict
import logging
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

class CitationAnalyzer:
    def __init__(self, data):
        self.data = data
        self.logger = logging.getLogger(__name__)

    def create_cocitation_network(self):
        """Create co-citation network from references"""
        self.logger.info("Creating co-citation network...")

        # Create a graph for co-citations
        G = nx.Graph()
        cocitation_counts = defaultdict(int)

        # Extract and process references
        processed_rows = 0
        for _, row in self.data.iterrows():
            try:
                if isinstance(row['References'], str):
                    # Split references and clean them
                    refs = [ref.strip() for ref in row['References'].split(';') if ref.strip()]
                    self.logger.debug(f"Processing {len(refs)} references from publication")

                    # Count co-citations
                    for i, ref1 in enumerate(refs):
                        for ref2 in refs[i+1:]:
                            if ref1 != ref2:
                                pair = tuple(sorted([ref1, ref2]))
                                cocitation_counts[pair] += 1

                processed_rows += 1
                if processed_rows % 10 == 0:
                    self.logger.debug(f"Processed {processed_rows} publications")

            except Exception as e:
                self.logger.error(f"Error processing row {processed_rows}: {str(e)}")
                continue

        # Add edges to graph based on co-citation frequency
        edge_count = 0
        for (ref1, ref2), count in cocitation_counts.items():
            if count > 1:  # Only include pairs cited together more than once
                G.add_edge(ref1, ref2, weight=count)
                edge_count += 1
                if edge_count % 1000 == 0:
                    self.logger.debug(f"Added {edge_count} edges to network")

        self.logger.info(f"Created co-citation network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G

    def cluster_cocitations(self, network, eps=0.3, min_samples=2):
        """Cluster co-citations using DBSCAN"""
        if network.number_of_nodes() < 2:
            self.logger.warning("Not enough nodes for clustering")
            return network

        self.logger.info("Starting co-citation clustering...")

        # Create normalized similarity matrix from network
        nodes = list(network.nodes())
        n = len(nodes)
        similarity_matrix = np.zeros((n, n))

        # Calculate similarities based on co-citation weights
        edge_weights = [d['weight'] for _,_,d in network.edges(data=True)]
        if not edge_weights:
            self.logger.warning("No edges found in network, skipping clustering")
            return network

        max_weight = max(edge_weights)
        self.logger.info(f"Maximum co-citation weight: {max_weight}")

        for i in range(n):
            for j in range(n):
                if i != j:
                    if network.has_edge(nodes[i], nodes[j]):
                        # Normalize weight to [0,1] range
                        similarity_matrix[i,j] = network[nodes[i]][nodes[j]]['weight'] / max_weight
                    else:
                        similarity_matrix[i,j] = 0

        # Log similarity matrix statistics
        self.logger.info(f"Similarity matrix stats - Min: {np.min(similarity_matrix)}, "
                        f"Max: {np.max(similarity_matrix)}, Mean: {np.mean(similarity_matrix)}")

        # Convert similarity to distance (1 - similarity)
        dist_matrix = 1 - similarity_matrix

        # Perform clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        clusters = clustering.fit_predict(dist_matrix)

        # Add cluster information to network
        cluster_counts = defaultdict(int)
        for node, cluster in zip(nodes, clusters):
            network.nodes[node]['cluster'] = int(cluster)
            cluster_counts[int(cluster)] += 1

        self.logger.info(f"Identified {len(set(clusters))} clusters")
        for cluster_id, count in cluster_counts.items():
            self.logger.debug(f"Cluster {cluster_id}: {count} nodes")

        return network

    def plot_cocitation_network(self, network, output_path):
        """Visualize co-citation network"""
        if network.number_of_nodes() == 0:
            self.logger.warning("Empty network, skipping visualization")
            return

        self.logger.info("Generating co-citation network visualization...")
        plt.figure(figsize=(15, 15))
        pos = nx.spring_layout(network, k=2, iterations=50)

        # Get cluster information
        clusters = nx.get_node_attributes(network, 'cluster')
        unique_clusters = set(clusters.values())
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))

        # Draw nodes colored by cluster
        for cluster_id, color in zip(sorted(unique_clusters), colors):
            nodelist = [node for node, c in clusters.items() if c == cluster_id]
            nx.draw_networkx_nodes(network, pos, nodelist=nodelist, 
                                 node_color=[color]*len(nodelist), alpha=0.6)

        # Draw edges with varying width based on weight
        edges = network.edges()
        weights = [network[u][v]['weight'] for u,v in edges]
        max_weight = max(weights) if weights else 1
        nx.draw_networkx_edges(network, pos, 
                             width=[w/max_weight * 2 for w in weights],
                             edge_color='gray', alpha=0.5)

        # Add labels
        nx.draw_networkx_labels(network, pos, font_size=8)

        plt.title('Co-Citation Network')
        plt.axis('off')

        # Save visualization
        self.logger.info(f"Saving co-citation network visualization to {output_path}")
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

    def export_cocitation_analysis(self, network, output_dir):
        """Export co-citation analysis results"""
        self.logger.info("Exporting co-citation analysis results...")

        try:
            # Export network metrics
            metrics = {
                'num_nodes': network.number_of_nodes(),
                'num_edges': network.number_of_edges(),
                'avg_cocitations': np.mean([d['weight'] for _,_,d in network.edges(data=True)]),
                'density': nx.density(network)
            }

            metrics_file = output_dir / 'cocitation_metrics.csv'
            pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
            self.logger.info(f"Exported network metrics to {metrics_file}")

            # Export edge list with weights
            edges = [(u, v, d['weight']) for u,v,d in network.edges(data=True)]
            edges_file = output_dir / 'cocitation_edges.csv'
            pd.DataFrame(edges, columns=['Source', 'Target', 'Weight'])\
              .to_csv(edges_file, index=False)
            self.logger.info(f"Exported edge list to {edges_file}")

        except Exception as e:
            self.logger.error(f"Error exporting co-citation analysis: {str(e)}")
            raise