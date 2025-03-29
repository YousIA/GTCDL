import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from models import SimpleNN, CNN, RNN, TransformerModel
from region_determination import region_determination
from knowledge_base import KnowledgeBase
from inference import infer
from offline_pruning import offline_model_pruning
from data_distribution import data_distribution_estimation
from game_theory import shapley_value

def generate_dummy_data(num_samples, num_features):
    X = np.random.randn(num_samples, num_features)
    y = np.random.randint(0, 2, num_samples)  # Binary classification
    return X, y

def main():
    num_samples = 1000
    num_features = 20
    num_classes = 2
    batch_size = 32
    num_epochs = 10

    X, y = generate_dummy_data(num_samples, num_features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = [
        SimpleNN(num_features, num_classes),
        CNN(1, num_classes),  # Assuming input channels = 1 for image-like data
        RNN(num_features, 64, 2, num_classes),
        TransformerModel(num_features, num_classes, num_heads=8, num_layers=2)
    ]

    loss_fn = nn.CrossEntropyLoss()

    losses, model_outputs = offline_model_pruning(models, torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long), loss_fn)
    
    distributions = data_distribution_estimation(X_train, time_window_size=10)
    avg_losses = calculate_average_loss(distributions, losses, models)
    
    knowledge_base = KnowledgeBase()
    for i, distribution in enumerate(distributions):
        distribution_info = {
            'mean': np.mean(distribution),
            'std': np.std(distribution),
            'avg_loss': {model: avg_losses[f'Model_{j}_Distribution_{i}'] for j, model in enumerate(models)}
        }
        knowledge_base.add_distribution(distribution_info)

    shapley_vals = shapley_value(models, losses, torch.tensor(y_train, dtype=torch.long))

    for i in range(len(X_test)):
        new_observation = torch.tensor(X_test[i], dtype=torch.float32)
        prediction = infer(new_observation, models, knowledge_base, shapley_vals)
        print(f"Prediction for sample {i}: {prediction}")

if __name__ == "__main__":
    main()
