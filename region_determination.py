import numpy as np
from scipy.spatial.distance import cdist

def initialize_centers(points, k):
    indices = np.random.choice(len(points), k, replace=False)
    return points[indices]

def assign_points_to_centers(points, centers):
    distances = cdist(points, centers)
    assignments = np.argmin(distances, axis=1)
    return assignments

def compute_max_diameter(points, assignments, k):
    max_diameter = 0
    for i in range(k):
        region_points = points[assignments == i]
        if len(region_points) > 1:
            region_diameter = np.max(cdist(region_points, region_points))
            max_diameter = max(max_diameter, region_diameter)
    return max_diameter

def region_determination(points, k, max_iter=100):
    centers = initialize_centers(points, k)
    assignments = assign_points_to_centers(points, centers)
    max_diameter = compute_max_diameter(points, assignments, k)

    for _ in range(max_iter):
        improved = False
        for i in range(k):
            region_points = points[assignments == i]
            if len(region_points) > 1:
                region_diameter = np.max(cdist(region_points, region_points))
                if region_diameter > max_diameter:
                    point_to_move = region_points[np.argmax(cdist(region_points, [centers[i]]))]
                    new_assignments = assign_points_to_centers(np.array([point_to_move]), centers)
                    new_region = new_assignments[0]
                    if new_region != i:
                        assignments[assignments == i] = new_region
                        centers[new_region] = point_to_move
                        max_diameter = compute_max_diameter(points, assignments, k)
                        improved = True
        if not improved:
            break

    return assignments, centers
