import itertools
import numpy as np

def coalition_value(coalition, losses, ground_truths):
    min_losses = np.min([losses[:, j] for j in coalition], axis=0)
    coalition_value = np.mean(min_losses)
    return coalition_value

def shapley_value(models, losses, ground_truths):
    n = len(models)
    shapley_values = np.zeros(n)

    for i in range(n):
        for subset in itertools.combinations(range(n), r=i):
            subset_with_i = subset + (i,)
            shapley_values[i] += (coalition_value(subset_with_i, losses, ground_truths) - 
                                  coalition_value(subset, losses, ground_truths)) / (len(subset) + 1)
        shapley_values[i] /= n

    return shapley_values

def banzhaf_value(models, losses, ground_truths):
    n = len(models)
    banzhaf_values = np.zeros(n)

    for i in range(n):
        for subset in itertools.combinations(range(n), r=i):
            subset_with_i = subset + (i,)
            banzhaf_values[i] += coalition_value(subset_with_i, losses, ground_truths) - coalition_value(subset, losses, ground_truths)
        banzhaf_values[i] /= 2 ** (n - 1)

    return banzhaf_values
