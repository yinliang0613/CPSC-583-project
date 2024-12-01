import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader

# Define the GAT model
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6):
        super(GAT, self).__init__()
        # First GAT layer with multi-head attention
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        # Second GAT layer (output layer) with a single attention head
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        # First GAT layer followed by ReLU activation
        x = self.conv1(x, edge_index).relu()
        # Apply dropout again after the first layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Second GAT layer (output layer)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Define the EGAT model
class EGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, mlp_hidden=16, heads=8, dropout=0.6, augmented_features=6):
        super(EGAT, self).__init__()
        # First GAT layer with multi-head attention
        self.conv1 = GATConv(in_channels-augmented_features, hidden_channels, heads=heads, dropout=dropout)
        # Second GAT layer (output layer) with a single attention head
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout
        self.augmented_features = augmented_features

        self.attention_mlp = torch.nn.Sequential(
            torch.nn.Linear(augmented_features, mlp_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_hidden, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x, edge_index):
        # Apply dropout to the input features
        original_features = x[:, :-self.augmented_features]
        appended_features = x[:, -self.augmented_features:]

        # Normalize the augmented features
        appended_features = (appended_features - appended_features.mean(dim=0)) / appended_features.std(dim=0)

        # get the attention scaling factor
        appended_scaling = self.attention_mlp(appended_features)

        # scale the original_features
        scaled_original_features = original_features * (1 + appended_scaling)

        x = scaled_original_features
        x = F.dropout(x, p=self.dropout, training=self.training)
        # First GAT layer followed by ReLU activation
        x = self.conv1(x, edge_index).relu()
        # Apply dropout again after the first layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Second GAT layer (output layer)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)