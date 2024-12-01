import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def train(model, data, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index)
    loss = loss_fn(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

@torch.no_grad()
def get_test_predictions(model, data):

    model.eval()
    output = model(data.x, data.edge_index)
    _, pred = output.max(dim=1)

    train_acc = (pred[data.train_mask] == data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()
    val_acc = (pred[data.val_mask] == data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
    test_acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()

    return data.y[data.test_mask].cpu().numpy(), pred[data.test_mask].cpu().numpy(), train_acc, val_acc, test_acc


def extract_augmented_features(data):
    # Assuming that augmented features are appended at the end of data.x
    closeness = data.x[:, -6].cpu().numpy()
    betweenness = data.x[:, -5].cpu().numpy()
    eigenvector = data.x[:, -4].cpu().numpy()
    degree = data.x[:, -3].cpu().numpy()
    harmonic = data.x[:, -2].cpu().numpy()
    page_rank = data.x[:, -1].cpu().numpy()
    return closeness, betweenness, eigenvector, degree, harmonic, page_rank



def plot_violin_plots(correct_features, incorrect_features, feature_name):
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        'Feature Value': np.concatenate([correct_features, incorrect_features]),
        'Prediction': ['Correct'] * len(correct_features) + ['Incorrect'] * len(incorrect_features)
    })
    print(df)

    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Prediction', y='Feature Value', data=df, palette={'Correct': 'green', 'Incorrect': 'red'})
    plt.title(f'Violin Plot of {feature_name}')
    plt.savefig(f'figures/violing_plot_{feature_name}.png')


