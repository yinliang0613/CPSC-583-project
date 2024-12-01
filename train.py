import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

import torch
from torch_geometric.datasets import Planetoid

# Import your model and utility functions here
from src.models import GAT, EGAT
from src.train_util import train, get_test_predictions, extract_augmented_features, plot_violin_plots
from src.data_util import augment_node_features

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = Planetoid(root=f'data/{args.dataset}', name=args.dataset)
    data = dataset[0].to(device)

    if args.model == "EGAT":
        data = augment_node_features(data)

    # Initialize lists to store metrics for each run
    precision_list = []
    recall_list = []
    f1_list = []
    train_acc_list = []
    val_acc_list = []
    test_acc_list = []

    with tqdm(range(args.n_runs)) as pbar:
        for i in pbar:
            # Initialize model, optimizer, and loss function
            if args.model == "GAT":
                model = GAT(dataset.num_features, hidden_channels=args.hidden_channels, out_channels=dataset.num_classes).to(device)
            elif args.model == "EGAT":
                model = EGAT(dataset.num_features, hidden_channels=args.hidden_channels, out_channels=dataset.num_classes).to(device)
            else:
                raise ValueError(f"Model {args.model} not supported")

            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            loss_fn = torch.nn.CrossEntropyLoss()

            # Training loop (simplified)
            model.train()
            with tqdm(range(1, args.epochs + 1), desc=f"Run {i+1}") as pbar:
                for epoch in pbar:
                  loss = train(model, data, optimizer, loss_fn)
                  pbar.set_postfix({'Loss': loss.item()})

            y_true, y_pred, train_acc, val_acc, test_acc = get_test_predictions(model, data)  # Implement this function

            # Calculate precision, recall, and F1-score for this run
            precision = precision_score(y_true, y_pred, average='macro')
            recall = recall_score(y_true, y_pred, average='macro')
            f1 = f1_score(y_true, y_pred, average='macro')

            # Store the metrics for this run
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)
            test_acc_list.append(test_acc)

    # Calculate mean and standard deviation across all runs
    precision_mean = np.mean(precision_list)
    precision_std = np.std(precision_list)

    recall_mean = np.mean(recall_list)
    recall_std = np.std(recall_list)

    f1_mean = np.mean(f1_list)
    f1_std = np.std(f1_list)

    train_acc_mean = np.mean(train_acc_list)
    train_acc_std = np.std(train_acc_list)

    val_acc_mean = np.mean(val_acc_list)
    val_acc_std = np.std(val_acc_list)

    test_acc_mean = np.mean(test_acc_list)
    test_acc_std = np.std(test_acc_list)

    print(f"Precision: {precision_mean:.4f} ± {precision_std:.4f}")
    print(f"Recall: {recall_mean:.4f} ± {recall_std:.4f}")
    print(f"F1-score: {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"Train accuracy: {train_acc_mean:.4f} ± {train_acc_std:.4f}")
    print(f"Validation accuracy: {val_acc_mean:.4f} ± {val_acc_std:.4f}")
    print(f"Test accuracy: {test_acc_mean:.4f} ± {test_acc_std:.4f}")

    # latex format
    print(
        f"{test_acc_mean * 100:.2f} $\pm$ {test_acc_std * 100:.2f} & "
        f"{precision_mean * 100:.2f} $\pm$ {precision_std * 100:.2f} & "
        f"{recall_mean * 100:.2f} $\pm$ {recall_std * 100:.2f} & "
        f"{f1_mean * 100:.2f} $\pm$ {f1_std * 100:.2f} & "
        f"{train_acc_mean * 100:.2f} $\pm$ {train_acc_std * 100:.2f} & "
        f"{val_acc_mean * 100:.2f} $\pm$ {val_acc_std * 100:.2f}")

    if args.plot:
        correct_predictions = (y_pred == y_true)
        incorrect_predictions = ~correct_predictions
        # Extract augmented features for all nodes
        closeness, betweenness, eigenvector, degree, harmonic, page_rank = extract_augmented_features(data)

        # Ensure tensors are on CPU before converting to numpy
        closeness = closeness[data.test_mask.cpu().numpy()]
        betweenness = betweenness[data.test_mask.cpu().numpy()]
        eigenvector = eigenvector[data.test_mask.cpu().numpy()]
        degree = degree[data.test_mask.cpu().numpy()]
        harmonic = harmonic[data.test_mask.cpu().numpy()]
        page_rank = page_rank[data.test_mask.cpu().numpy()]

        # Plot violin plots for each feature
        plot_violin_plots(closeness[correct_predictions], closeness[incorrect_predictions], 'Closeness Centrality')
        plot_violin_plots(betweenness[correct_predictions], betweenness[incorrect_predictions], 'Betweenness Centrality')
        plot_violin_plots(eigenvector[correct_predictions], eigenvector[incorrect_predictions], 'Eigenvector Centrality')
        plot_violin_plots(degree[correct_predictions], degree[incorrect_predictions], 'Degree Centrality')
        plot_violin_plots(harmonic[correct_predictions], harmonic[incorrect_predictions], 'Harmonic Centrality')
        plot_violin_plots(page_rank[correct_predictions], page_rank[incorrect_predictions], 'PageRank')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--hidden_channels",
        type=int,
        default=200,
        help="Number of hidden channels in the model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=10,
        help="Number of runs to average over"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Cora",
        help="Name of the dataset to use (Cora, Citeseer, Pubmed)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="EGAT",
        help="Name of the model to use (GAT, EGAT)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Whether to plot the violin plots"
    )
    main(parser.parse_args())