from collections import deque
import copy
import random
from typing import NamedTuple, Tuple, Union

from model import Actor, Critic
import numpy as np
import torch 
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of actor
LR_CRITIC = 1e-3        # learning rate of critic
WEIGHT_DECAY = 0        # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    """
    Interact with and learn from the environment
    """
    def __init__(
        self, 
        state_size: int, 
        action_size: int,
        random_seed: int,
    ) -> None:
        """
        Initialize agent.

        Params
        ======
        state_size: dimension of state
        action_size: dimension of action
        random_seed: int
        """
        self.state_size = state_size 
        self.action_size = action_size

        # Actor 
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise 
        self.noise = OUNoise(action_size)

        # Replay buffer
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)

    def step(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool,
        update: bool,
    ) -> None:
        """
        Save experience and learn.

        Params
        ======
        state: current state
        action: action taken
        reward: reward as a result of the action
        next_state: next state as a result of the action
        done: True if episode is done
        update: update networks if True
        """
        # Save experience in replay memory.
        self.memory.add(state, action, reward, next_state, done)

        # Learn if there are enough samples in memory.
        if update and len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state: np.ndarray, add_noise: bool = True) -> int:
        """
        Return action given the state.

        Params
        ======
        state: current state
        add_noise: whether to add random noise for exploration

        Returns
        =======
        action
        """
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self) -> None:
        """
        Reset action noise.
        """
        self.noise.reset()

    def learn(
        self, 
        experiences: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], 
        gamma: float,
    ) -> None:
        """
        Update network parameters based on batch of experiences.

        Params
        ======
        experiences = tuple of (state, action, reward, next_state, done)
        gamma: discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ----------------------- Update critic -------------------------------#
        # Get predicted next state actions and Q values from target models.
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states.
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss.
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize loss.
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ----------------------- Update actor --------------------------------#       
        # Compute actor loss.
        actions_pred = self.actor_local(states)
        actor_loss = - self.critic_local(states, actions_pred).mean()
        # Minimize loss.
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------------------- Update target networks --------------------------#
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(
        self, 
        local_model: Union[Actor, Critic], 
        target_model: Union[Actor, Critic], 
        tau: float
    ) -> None:
        """
        Soft update model parameters.

        Params
        ======
        local_model: model from which weights will be copied.
        target_model: model to which weights will be copied.
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        

class OUNoise:
    """
    Ornstein-Uhlenbeck process to add randomness (exploration) to action selection.
    """
    def __init__(
        self, 
        size: int, 
        mu: float = 0., 
        theta: float = 0.15, 
        sigma: float = 0.1,
    ) -> None:
        """
        Initialize parameters and noise process.

        Params
        ======
        size: dimension of the action noise.
        random_seed
        mu, theta, sigma: parameters of Ornstein-Uhlenbeck process.
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self) -> None:
        """
        Reset the internal state (noise) to mean (mu).
        """
        self.state = copy.copy(self.mu)

    def sample(self) -> np.ndarray:
        """
        Update the internal state and return it as a noise sample.

        Returns
        =======
        noise array
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.random(size=len(x))
        self.state = x + dx 
        return self.state


class Experience(NamedTuple):
    """
    Experience tuple.
    """
    state: np.ndarray 
    action: int
    reward: float 
    next_state: np.ndarray 
    done: bool


class ReplayBuffer:
    """
    Buffer to store experience tuples.
    """
    def __init__(self, buffer_size: int, batch_size: int) -> None:
        """
        Initialize buffer.

        Params
        ======
        buffer_size: max size of buffer
        batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size 

    def add(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ) -> None:
        """
        Add new experience to memory buffer.

        Params
        ======
        state: current state
        action: action taken
        reward: reward due to action
        next_state: next state due to action
        done: whether episode is done
        """
        self.memory.append(Experience(state, action, reward, next_state, done))

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly sample a batch of experiences from memory.

        Returns
        =======
        tuple of (states, actions, rewards, next_states, dones)
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None])).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self) -> int:
        """
        Get size of memory buffer.

        Returns
        =======
        Number of experience tuples in the memory.
        """
        return len(self.memory)
