import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Sequence.sequence_model import SequenceGameRule
import os
import time

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.qnetwork_local = QNetwork(state_size, action_size, seed)
        self.qnetwork_target = QNetwork(state_size, action_size, seed)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=0.001)

        self.memory = []
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 0.99
        self.epsilon_min = 0.05  # Increased from 0.01 to 0.05
        self.epsilon_decay = 0.99
        self.t_step = 0
        self.update_freq = 4

    def step(self, state, action_idx, reward, next_state, done, legal_actions, next_legal_actions):
        self.memory.append((state, action_idx, reward, next_state, done, legal_actions, next_legal_actions))
        self.t_step = (self.t_step + 1) % self.update_freq
        if self.t_step == 0 and len(self.memory) >= self.batch_size:
            self._learn()

    def act(self, state, legal_actions, game_action_to_idx_map, idx_to_game_action_map):
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > self.epsilon:
            candidate_actions = []
            for action in legal_actions:
                action_key = frozenset(action.items())
                if action_key in game_action_to_idx_map:
                    idx = game_action_to_idx_map[action_key]
                    q_value = action_values[0][idx].item()
                    candidate_actions.append((q_value, idx))
            if candidate_actions:
                candidate_actions.sort(key=lambda x: x[0], reverse=True)
                return candidate_actions[0][1]
        return get_or_create_action_idx(random.choice(legal_actions))

    def _learn(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long()
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float()
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float()
        legal_actions_list = [e[5] for e in experiences]
        next_legal_actions_list = [e[6] for e in experiences]

        self.qnetwork_target.eval()
        with torch.no_grad():
            next_q_values = self.qnetwork_target(next_states)
        Q_targets_next = torch.zeros_like(next_q_values)
        for i in range(self.batch_size):
            next_legal_actions = next_legal_actions_list[i]
            for action in next_legal_actions:
                action_key = frozenset(action.items())
                idx = game_action_to_idx_map.get(action_key, -1)
                if idx != -1 and idx < next_q_values.shape[1]:
                    Q_targets_next[i][idx] = next_q_values[i][idx]
        Q_targets_next = Q_targets_next.max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._soft_update(self.qnetwork_local, self.qnetwork_target, 0.01)

    def _soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# Initialize game and agent
state_size = 10
action_size = 5000
seed = 0
agent = DQNAgent(state_size, action_size, seed)
game_rule = SequenceGameRule(num_of_agent=2)

# Action mapping
game_action_to_idx_map = {}
idx_to_game_action_map = {}
next_action_idx = 0

def get_or_create_action_idx(action_dict):
    global next_action_idx
    action_key = frozenset(action_dict.items())
    if action_key in game_action_to_idx_map:
        return game_action_to_idx_map[action_key]
    if next_action_idx < action_size:
        game_action_to_idx_map[action_key] = next_action_idx
        idx_to_game_action_map[next_action_idx] = action_dict
        idx = next_action_idx
        next_action_idx += 1
        return idx
    return random.randint(0, action_size - 1)

# Convert state to vector (must match myTeam.py)
def convert_state_to_vector(game_state, legal_actions, agent_id=0):
    num_legal_actions = len(legal_actions)
    num_place_actions = sum(1 for action in legal_actions if action['type'] == 'place')
    num_remove_actions = sum(1 for action in legal_actions if action['type'] == 'remove')
    own_seqs = game_state.agents[agent_id].completed_seqs
    opp_id = 1 - agent_id
    opp_seqs = game_state.agents[opp_id].completed_seqs
    hand_size = len(game_state.agents[agent_id].hand)
    num_jacks = sum(1 for card in game_state.agents[agent_id].hand if card is not None and 'Jack' in card)
    num_two_eyed_jacks = sum(1 for card in game_state.agents[agent_id].hand if card is not None and 'Two-Eyed Jack' in card)
    num_one_eyed_jacks = sum(1 for card in game_state.agents[agent_id].hand if card is not None and 'One-Eyed Jack' in card)
    central_squares = [(4,4), (4,5), (5,4), (5,5)]
    num_central_positions = sum(1 for r in range(10) for c in range(10) if (r, c) in central_squares and game_state.board.chips[r][c] == agent_id)
    state_vector = np.array([num_legal_actions, num_place_actions, num_remove_actions,
                             own_seqs, opp_seqs, hand_size, num_jacks, num_two_eyed_jacks, num_one_eyed_jacks, num_central_positions], dtype=np.float32)
    return state_vector

# Training loop
n_episodes = 500
max_steps = 150  # Reduced from 200 to 150

for e in range(n_episodes):
    state = game_rule.initialGameState()
    legal_actions = game_rule.getLegalActions(state, 0)
    state_vector = convert_state_to_vector(state, legal_actions)
    scores = {0: 0, 1: 0}
    steps = 0
    start_time = time.time()

    while steps < max_steps:
        # Agent 0 (DQN)
        action_idx = agent.act(state_vector, legal_actions, game_action_to_idx_map, idx_to_game_action_map)
        action = idx_to_game_action_map.get(action_idx, random.choice(legal_actions))
        next_state = game_rule.generateSuccessor(state, action, 0)
        next_legal_actions = game_rule.getLegalActions(next_state, 1)
        next_state_vector = convert_state_to_vector(next_state, next_legal_actions, 0)

        # Reward
        reward = 0
        if next_state.agents[0].completed_seqs > state.agents[0].completed_seqs:
            reward += 2000  # Increased from 1000 to 2000
        if next_state.agents[1].completed_seqs > state.agents[1].completed_seqs:
            reward -= 1500  # Increased from -800 to -1500
        if 'play_card' in action and action['play_card'] and 'Jack' in action['play_card']:
            reward += 50
        if 'draft_card' in action and action['draft_card'] and 'Jack' in action['draft_card']:
            reward += 30

        done = next_state.agents[0].completed_seqs >= 2 or next_state.agents[1].completed_seqs >= 2 or not next_legal_actions
        agent.step(state_vector, action_idx, reward, next_state_vector, done, legal_actions, next_legal_actions)

        state = next_state
        state_vector = next_state_vector
        legal_actions = next_legal_actions
        steps += 1

        # Agent 1 (Random)
        if not done:
            legal_actions = game_rule.getLegalActions(state, 1)
            if not legal_actions:
                break
            action = random.choice(legal_actions)
            state = game_rule.generateSuccessor(state, action, 1)
            legal_actions = game_rule.getLegalActions(state, 0)
            state_vector = convert_state_to_vector(state, legal_actions)
            steps += 1

        scores[0] = state.agents[0].completed_seqs
        scores[1] = state.agents[1].completed_seqs
        done = scores[0] >= 2 or scores[1] >= 2 or not legal_actions

        if done:
            break

    agent.decay_epsilon()
    termination_reason = "Unknown"
    if scores[0] >= 2:
        termination_reason = "RED won with 2 sequences"
    elif scores[1] >= 2:
        termination_reason = "BLU won with 2 sequences"
    elif not legal_actions:
        termination_reason = "No legal actions left"
    elif steps >= max_steps:
        termination_reason = "Max steps reached"

    print(f"Episode {e+1}/{n_episodes}, Steps: {steps}, Scores: RED={scores[0]}, BLU={scores[1]}, Epsilon: {agent.epsilon:.2f}, Termination: {termination_reason}")

# Save the model
save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 't_094'))
os.makedirs(save_dir, exist_ok=True)
local_path = os.path.join(save_dir, "hybrid_sequence_agent_local.pth")
target_path = os.path.join(save_dir, "hybrid_sequence_agent_target.pth")
torch.save(agent.qnetwork_local.state_dict(), local_path)
torch.save(agent.qnetwork_target.state_dict(), target_path)
print(f"Training complete. Model weights saved to {save_dir}.")