Game-Theoretic Consensus Deep Learning for Adaptive Flood Prediction in Digital Twin Environments
The framework includes multiple deep learning models, leverages game-theoretic concepts like Shapley Value for model importance, and uses a two-level pruning strategy for efficient inference.

The project is organized into the following files:

- models.py: Contains four different neural network models for classification:
  - Simple Feedforward Neural Network
  - Convolutional Neural Network (CNN)
  - Recurrent Neural Network (RNN)
  - Transformer Model

- offline_pruning.py: Implements the first-level offline model pruning based on loss values.

- data_distribution.py: Handles data distribution estimation from the training data.

- game_theory.py: Includes functions for game-theoretic learning, such as calculating Shapley Value for model importance.

- region_determination.py: Implements the second-level online model pruning using region determination.

- knowledge_base.py: Manages the creation and retrieval of knowledge base information.

- inference.py: Contains the inference logic, including weighted voting based on model importance.

- main.py: The main script that integrates all components to run the entire process from training to inference.



Install the required dependencies: pip install -r requirements.txt

To execute the entire process, including training, pruning, and inference:

python main.py
