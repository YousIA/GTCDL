import numpy as np

def weighted_voting(models, observation, best_distribution, shapley_values):
    weighted_sum = 0
    total_weight = 0

    for model, shapley_value in zip(models, shapley_values):
        prediction = model(observation).detach().numpy()
        weight = best_distribution['avg_loss'][model] / shapley_value
        weighted_sum += weight * prediction
        total_weight += weight

    return weighted_sum / total_weight

def infer(observation, models, knowledge_base, shapley_values):
    best_distribution = knowledge_base.find_best_distribution(observation)
    return weighted_voting(models, observation, best_distribution, shapley_values)
