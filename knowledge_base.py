import numpy as np

class KnowledgeBase:
    def __init__(self):
        self.distributions = []

    def add_distribution(self, distribution_info):
        self.distributions.append(distribution_info)

    def find_best_distribution(self, observation):
        best_match = None
        min_distance = float('inf')
        for dist in self.distributions:
            mean_diff = np.linalg.norm(np.mean(observation) - dist['mean'])
            std_diff = np.linalg.norm(np.std(observation) - dist['std'])
            total_diff = mean_diff + std_diff
            if total_diff < min_distance:
                min_distance = total_diff
                best_match = dist
        return best_match
