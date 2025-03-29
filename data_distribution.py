import numpy as np

def calculate_probability_distribution(values, window_size):
    unique, counts = np.unique(values, return_counts=True)
    probabilities = counts / window_size
    return dict(zip(unique, probabilities))

def data_distribution_estimation(observations, time_window_size):
    distributions = []
    for i in range(0, len(observations), time_window_size):
        window = observations[i:i+time_window_size]
        distribution = calculate_probability_distribution(window, time_window_size)
        distributions.append(distribution)
    return distributions

def calculate_average_loss(distributions, losses, models):
    avg_losses = {}
    for i, distribution in enumerate(distributions):
        for j, model in enumerate(models):
            avg_loss = np.mean([losses[k, j] for k, _ in enumerate(distribution)])
            avg_losses[f'Model_{j}_Distribution_{i}'] = avg_loss
    return avg_losses
