import torch
from torch_geometric.utils import degree
import networkx as nx
from torch_geometric.utils import to_networkx

def get_n_neighbors(graph, idx):
  n_neighbors = graph.edge_index[0].tolist().count(idx)
  return n_neighbors

def get_num_classes(data):
  num_classes = data.y.unique().shape[0]
  return num_classes

def get_idx_with_most_neighbors(data):
  neighbors = [get_n_neighbors(data, i) for i in range(data.num_nodes)]
  idx = neighbors.index(max(neighbors))
  return idx

def augment_node_features(data):
    # Convert PyTorch Geometric graph to NetworkX graph for centrality calculations
    G = to_networkx(data, to_undirected=True)

    # Compute centrality measures using NetworkX
    closeness_centrality = nx.closeness_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G)


    # Additional centrality measures
    degree_centrality = nx.degree_centrality(G)
    harmonic_centrality = nx.harmonic_centrality(G)
    pagerank = nx.pagerank(G)

     # Create feature vectors for each node based on centralities
    closeness = torch.tensor([closeness_centrality[i] for i in range(data.num_nodes)], dtype=torch.float).unsqueeze(1)
    betweenness = torch.tensor([betweenness_centrality[i] for i in range(data.num_nodes)], dtype=torch.float).unsqueeze(1)
    eigenvector = torch.tensor([eigenvector_centrality[i] for i in range(data.num_nodes)], dtype=torch.float).unsqueeze(1)

    # Additional features
    degree = torch.tensor([degree_centrality[i] for i in range(data.num_nodes)], dtype=torch.float).unsqueeze(1)
    harmonic = torch.tensor([harmonic_centrality[i] for i in range(data.num_nodes)], dtype=torch.float).unsqueeze(1)
    page_rank = torch.tensor([pagerank[i] for i in range(data.num_nodes)], dtype=torch.float).unsqueeze(1)

    # Concatenate these centralities to the original node features
    data.x = torch.cat([
        data.x, closeness, betweenness, eigenvector,
        degree, harmonic, page_rank
    ], dim=1)

    return data