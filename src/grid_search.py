import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
from torch_geometric.datasets import Planetoid

from src.models import EGAT
from src.data_util import get_num_classes, augment_node_features
from src.train_util import train, get_test_predictions


def main(args):
    learning_rates = [1e-3, 1e-4, 1e-5, 1e-6]
    epochs_list = [10, 20, 40, 60, 80, 100]
    optimizers = [optim.Adam, optim.SGD]  # Add more optimizers as needed
    optimizer_names = ['Adam', 'SGD']  # Match optimizer list


    columns = ['Learning Rate', 'Epochs', 'Optimizer', 'Final Test Accuracy']
    results_df = pd.DataFrame(columns=columns)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = Planetoid(root=f'data/{args.dataset}', name=args.dataset)
    data = dataset[0].to(device)
    data = augment_node_features(data)

    # Assuming data is already prepared and loaded
    num_features = data.num_features
    num_classes = get_num_classes(data)

    for lr in learning_rates:
        for epochs in tqdm(epochs_list):
            for opt, opt_name in zip(optimizers, optimizer_names):
                model = EGAT(num_features, args.hidden_channels, num_classes)
                loss_fn = nn.CrossEntropyLoss()
                best_val_acc = final_test_acc = 0

                ############# Your code here ############
                ## (~7 line of code)
                optimizer = opt(model.parameters(), lr=lr)
                for epoch in tqdm(range(1, epochs + 1)):
                    loss = train(model, data, optimizer, loss_fn)
                    _, _, train_acc, val_acc, test_acc = get_test_predictions(model, data)
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        final_test_acc = test_acc
                #########################################

                # Store the results in the DataFrame
                new_row = pd.DataFrame({'Learning Rate': [lr],
                                        'Epochs': [epochs],
                                        'Optimizer': [opt_name],
                                        'Final Test Accuracy': [final_test_acc]})

                results_df = pd.concat([results_df, new_row], ignore_index=True)


    # Plotting accuracy vs. epochs for each learning rate and optimizer
    for lr in learning_rates:
        for opt_name in optimizer_names:
            subset_df = results_df[(results_df['Learning Rate'] == lr) & (results_df['Optimizer'] == opt_name)]
            plt.plot(subset_df['Epochs'], subset_df['Final Test Accuracy'], label=f'{opt_name}, LR={lr}')

    plt.xlabel('Epochs')
    plt.ylabel('Final Test Accuracy')
    plt.legend()
    plt.show()
    print(f'Best test accuracy: {results_df["Final Test Accuracy"].max()} is achieved with the following parameters:')
    print(results_df[results_df["Final Test Accuracy"] == results_df["Final Test Accuracy"].max()])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hidden_channels",
        type=int,
        default=200,
        help="Number of hidden channels in the GAT model"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Cora",
        help="Name of the dataset to use (Cora, Citeseer, Pubmed)"
    )
    main(parser.parse_args())