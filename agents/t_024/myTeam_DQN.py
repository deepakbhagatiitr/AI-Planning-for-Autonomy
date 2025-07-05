import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from copy import deepcopy
import os  
from template import Agent

from Sequence.sequence_model import SequenceGameRule  
from Sequence.sequence_utils import EMPTY, JOKER, RED, BLU, RED_SEQ, BLU_SEQ  
# --- DQN Components ---

# Q-Network Definition
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


# Experience Replay Buffer
Experience = namedtuple("Experience", field_names=["state", "action_map_idx", "reward", "next_state", "done"])


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        random.seed(seed)

    def add(self, state, action_map_idx, reward, next_state, done):
        e = Experience(state, action_map_idx, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        action_map_idxs = torch.from_numpy(np.vstack([e.action_map_idx for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()

        return (states, action_map_idxs, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


class myAgent(Agent):  # Renamed to myAgent for compatibility with general_game_runner.py
    BOARD_ROWS = 10
    BOARD_COLS = 10
    BOARD_CELL_STATES = 6

    
    ALL_RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
    ALL_SUITS = ['S', 'H', 'D', 'C']  # Spades, Hearts, Diamonds, Clubs

    CARD_STRINGS = []
    for _ in range(2):  # Two decks
        for suit in ALL_SUITS:
            for rank in ALL_RANKS:
                CARD_STRINGS.append(rank + suit)


    CARD_TO_INT_MAP = {card_str: i for i, card_str in
                       enumerate(sorted(list(set(CARD_STRINGS))))}  # Ensure unique mapping
    VOCAB_SIZE = len(CARD_TO_INT_MAP)
    MAX_HAND_SIZE = 7
    ACTION_SPACE_SIZE_FOR_NETWORK = 1000  # Max unique (action dicts) agent can map

    # Default model path (relative to where the agent script is run)
    DEFAULT_MODEL_LOAD_PREFIX = "dqn_sequence_agent"

    def __init__(self, _id, num_agents=2, seed=0, training_mode=False, model_load_prefix=None,
                 buffer_size=int(1e4), batch_size=64,
                 gamma=0.99, tau=1e-3, lr=5e-4, update_every=4):
        super().__init__(_id)
        self.id = _id
        self.num_agents = num_agents
        self.training_mode = training_mode

        board_flat_size = self.BOARD_ROWS * self.BOARD_COLS * self.BOARD_CELL_STATES
        hand_vector_size = self.VOCAB_SIZE
        other_features_size = 3  # my_score_norm, opp_score_norm, turn_phase_info (e.g. can_trade)
        self.state_size = board_flat_size + hand_vector_size + other_features_size

        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.game_action_to_idx_map = {}
        self.idx_to_game_action_map = {}
        self.next_action_idx = 0

        self.qnetwork_local = QNetwork(self.state_size, self.ACTION_SPACE_SIZE_FOR_NETWORK, seed)
        self.qnetwork_target = QNetwork(self.state_size, self.ACTION_SPACE_SIZE_FOR_NETWORK, seed)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size, batch_size, seed)

        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.t_step = 0

        if self.training_mode:
            self.epsilon = 1.0
            self.epsilon_decay = 0.995
            self.epsilon_min = 0.01
        else:
            self.epsilon = 0.05  # Low epsilon for evaluation (some exploration if policy is poor)
            self.epsilon_decay = 1.0  # No decay in eval mode
            self.epsilon_min = 0.05

        self._current_chosen_action_map_idx = None

        # Attempt to load model
        load_prefix = model_load_prefix if model_load_prefix is not None else self.DEFAULT_MODEL_LOAD_PREFIX
        if not self.training_mode:  # Or always try to load if a model is available
            self.load_model(load_prefix)

    def _convert_state_to_vector(self, game_state):
        board_features = np.zeros((self.BOARD_ROWS, self.BOARD_COLS, self.BOARD_CELL_STATES), dtype=np.float32)
        my_agent_state = game_state.agents[self.id]
        opp_agent_id = (self.id + 1) % self.num_agents
        opp_agent_state = game_state.agents[opp_agent_id]


        my_colour_chip = RED if my_agent_state.colour == RED else BLU  # Example
        opp_colour_chip = BLU if my_colour_chip == RED else RED  # Example
        my_seq_chip_type = RED_SEQ if my_colour_chip == RED else BLU_SEQ  # Example
        opp_seq_chip_type = BLU_SEQ if my_colour_chip == RED else RED_SEQ  # Example

        for r in range(self.BOARD_ROWS):
            for c in range(self.BOARD_COLS):
                chip_on_board = game_state.board.chips[r][c]
                if chip_on_board == EMPTY:
                    board_features[r, c, 0] = 1.0
                elif chip_on_board == my_colour_chip:
                    board_features[r, c, 1] = 1.0
                elif chip_on_board == opp_colour_chip:
                    board_features[r, c, 2] = 1.0
                elif chip_on_board == my_seq_chip_type:
                    board_features[r, c, 3] = 1.0
                elif chip_on_board == opp_seq_chip_type:
                    board_features[r, c, 4] = 1.0
                elif chip_on_board == JOKER:  # Corner JOKER spaces
                    board_features[r, c, 5] = 1.0
        board_flat = board_features.flatten()

        hand_vector = np.zeros(self.VOCAB_SIZE, dtype=np.float32)
        for card_str in my_agent_state.hand:
            if card_str in self.CARD_TO_INT_MAP:  # Robust check
                hand_vector[self.CARD_TO_INT_MAP[card_str]] += 1.0

        my_score_norm = my_agent_state.score / 2.0
        opp_score_norm = opp_agent_state.score / 2.0

        cards_in_hand_norm = len(my_agent_state.hand) / self.MAX_HAND_SIZE if self.MAX_HAND_SIZE > 0 else 0.0

        state_vector = np.concatenate((board_flat, hand_vector,
                                       [my_score_norm, opp_score_norm, cards_in_hand_norm])).astype(np.float32)

        if state_vector.shape[0] != self.state_size:
            raise ValueError(
                f"State vector length mismatch: expected {self.state_size}, got {state_vector.shape[0]}. Board: {board_flat.shape}, Hand: {hand_vector.shape}, Vocab: {self.VOCAB_SIZE}")
        return state_vector

    def _get_or_create_action_map_idx(self, game_action_dict):
        action_key = frozenset(game_action_dict.items())

        if action_key in self.game_action_to_idx_map:
            return self.game_action_to_idx_map[action_key]
        elif self.next_action_idx < self.ACTION_SPACE_SIZE_FOR_NETWORK:
            current_idx = self.next_action_idx
            self.game_action_to_idx_map[action_key] = current_idx
            self.idx_to_game_action_map[current_idx] = game_action_dict
            self.next_action_idx += 1
            return current_idx
        else:

            return None

    def step(self, state_vector, action_map_idx, reward, next_state_vector, done):
        if not self.training_mode:  # Do not learn or store if not in training mode
            return
        if action_map_idx is None:
            return

        self.memory.add(state_vector, action_map_idx, reward, next_state_vector, done)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def SelectAction(self, legal_game_actions, game_state):
        if not legal_game_actions:  # No legal actions
            return None

        state_vector = self._convert_state_to_vector(game_state)
        state_tensor = torch.from_numpy(state_vector).float().unsqueeze(0)

        self.qnetwork_local.eval()
        with torch.no_grad():
            all_q_values_for_network_actions = self.qnetwork_local(state_tensor).squeeze(0)
        if self.training_mode:  # Set back to train mode only if actually training
            self.qnetwork_local.train()

        chosen_game_action = None
        self._current_chosen_action_map_idx = None

        if random.random() > self.epsilon:  # Exploit
            best_q_value = -float('inf')
            candidate_actions = []  # Store (q_value, game_action_dict)

            for game_action_dict in legal_game_actions:
                action_map_idx = self._get_or_create_action_map_idx(game_action_dict)

                if action_map_idx is not None:
                    q_value = all_q_values_for_network_actions[action_map_idx].item()
                    candidate_actions.append((q_value, game_action_dict, action_map_idx))
                else:
                    # This action is new and unmappable (limit reached).
                    # We can't get a Q-value for it from the network.
                    # For exploitation, we should ideally not pick it unless no other options.
                    pass  # Or assign a very low Q-value for consideration

            if candidate_actions:
                candidate_actions.sort(key=lambda x: x[0], reverse=True)  # Sort by Q-value descending
                best_q_value, chosen_game_action, self._current_chosen_action_map_idx = candidate_actions[0]

            if chosen_game_action is None:  # Fallback: if all legal actions were new and unmappable
                # print("Warning: Exploitation chose random due to all legal actions being unmappable or no candidates.")
                chosen_game_action = random.choice(legal_game_actions)
                self._current_chosen_action_map_idx = self._get_or_create_action_map_idx(
                    chosen_game_action)  # Try to map the random one
        else:  # Explore
            chosen_game_action = random.choice(legal_game_actions)
            self._current_chosen_action_map_idx = self._get_or_create_action_map_idx(chosen_game_action)

        # Decay epsilon if in training mode
        if self.training_mode:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return chosen_game_action

    def get_chosen_action_map_idx(self):
        return self._current_chosen_action_map_idx

    def learn(self, experiences, gamma):
        states, action_map_idxs, rewards, next_states, dones = experiences

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, action_map_idxs)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def _soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save_model(self, filename_prefix="dqn_sequence_agent"):
        """Saves the local Q-network's state dictionary."""
        try:
            path_local = f"{filename_prefix}_local.pth"
            torch.save(self.qnetwork_local.state_dict(), path_local)
            # print(f"Local Q-network saved to {path_local}")
            # Target network is just a delayed copy, usually not saved separately unless for specific resume strategies
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, filename_prefix="dqn_sequence_agent"):
        """Loads the Q-network's state dictionary for both local and target networks."""
        try:
            path_local = f"{filename_prefix}_local.pth"
            if os.path.exists(path_local):
                self.qnetwork_local.load_state_dict(torch.load(path_local))
                self.qnetwork_target.load_state_dict(torch.load(path_local))  # Initialize target same as local
                self.qnetwork_local.train() if self.training_mode else self.qnetwork_local.eval()
                self.qnetwork_target.eval()
                # print(f"Models loaded from {path_local}")
            else:
                # print(f"Model file not found: {path_local}. Starting with new model.")
                pass
        except Exception as e:
            print(f"Error loading model: {e}. Starting with new model.")