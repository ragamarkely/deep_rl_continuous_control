# Project 2: Continuous Control

## Implementation
In this project, [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/abs/1509.02971) algorithm is used to solve the Reacher environment with 20 agents.

Each of the Actor's and Critic's local and target networks is a Deep Neural Network with 2 hidden layers of fully connected networks. Each of the hidden layers has 128 units. In addition, each of these networks uses batch normalization to scale the output of the input layers. A gradient clipping is also used in training the Critic's network. One of the most important element of the implementation that significantly help stabilize and accelerate the learning is the scheduling of network updates. In this implementation, the networks are updated 10 times after every 20 time steps. If the networks are updated too frequently, the learning becomes unstable and very slow.

## Future Work
The main challenge in this project is that the performance of DDPG agent is sensitive to hyperparameter changes, especially when the networks are updated too frequently. There are several analysis that can be done to explored to further improve the model performance, such as:
- The impact of varying the number of updates and time intervals between the updates on the speed as well as stability of the learning.
- The impact of increasing the model complexity, such as larger number of units and hidden layers on model training time and performance. We can also study whether the number of updates and time intervals between updates need to be changed when larger networks are used.
- Explore other Actor Critic algorithms, such as A2C, A3C, and GAE for better performance.