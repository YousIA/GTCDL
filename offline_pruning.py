import torch
import torch.nn as nn
import numpy as np

def calculate_loss(model, observation, ground_truth, loss_fn):
    prediction = model(observation)
    loss = loss_fn(prediction, ground_truth)
    return loss.item()

def get_model_outputs(models, observations):
    model_outputs = {}
    for i, model in enumerate(models):
        outputs = []
        for observation in observations:
            output = model(observation).detach().numpy()
            outputs.append(output)
        model_outputs[f'Model_{i}'] = np.concatenate(outputs)
    return model_outputs

def offline_model_pruning(models, observations, ground_truths, loss_fn):
    losses = np.zeros((len(observations), len(models)))

    for i, observation in enumerate(observations):
        for j, model in enumerate(models):
            losses[i, j] = calculate_loss(model, observation, ground_truths[i], loss_fn)
    
    model_outputs = get_model_outputs(models, observations)

    return losses, model_outputs
