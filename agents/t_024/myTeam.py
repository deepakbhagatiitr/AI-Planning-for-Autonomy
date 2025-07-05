import sys
import os
import random
import time
from Sequence.sequence_model import SequenceGameRule, RED, BLU
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

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

class HybridAgent:
    THINKTIME = 0.95
    ACTION_SPACE_SIZE_FOR_NETWORK = 5000
    DEFAULT_MODEL_LOAD_PREFIX = "hybrid_sequence_agent"

    def __init__(self, _id, num_agents=2, seed=0, model_load_prefix=None):
        print(f"Starting initialization of HybridAgent (Agent ID: {_id})")
        self.id = _id
        self.num_agents = num_agents
        self.colour = RED if self.id == 0 else BLU

        # DQN State Representation
        self.state_size = 10

        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.game_action_to_idx_map = {}
        self.idx_to_game_action_map = {}
        self.next_action_idx = 0
        self.qnetwork_local = QNetwork(self.state_size, self.ACTION_SPACE_SIZE_FOR_NETWORK, seed)
        self.qnetwork_target = QNetwork(self.state_size, self.ACTION_SPACE_SIZE_FOR_NETWORK, seed)
        self.epsilon = 0.05

        self._current_chosen_action_map_idx = None

        self.game_rule = SequenceGameRule(num_of_agent=self.num_agents)
        self.central_squares = [(4,4), (4,5), (5,4), (5,5)]
        self.key_corners = [(0,0), (0,9), (9,0), (9,9)]

        load_prefix = model_load_prefix if model_load_prefix is not None else self.DEFAULT_MODEL_LOAD_PREFIX
        self.load_model(load_prefix)
        print(f"Completed initialization of HybridAgent (Agent ID: {_id})")

    def _convert_state_to_vector(self, game_state, legal_actions):
        num_legal_actions = len(legal_actions)
        num_place_actions = sum(1 for action in legal_actions if action['type'] == 'place')
        num_remove_actions = sum(1 for action in legal_actions if action['type'] == 'remove')
        own_seqs = game_state.agents[self.id].completed_seqs
        opp_id = 1 - self.id
        opp_seqs = game_state.agents[opp_id].completed_seqs
        hand_size = len(game_state.agents[self.id].hand)
        num_jacks = sum(1 for card in game_state.agents[self.id].hand if card is not None and 'Jack' in card)
        num_two_eyed_jacks = sum(1 for card in game_state.agents[self.id].hand if card is not None and 'Two-Eyed Jack' in card)
        num_one_eyed_jacks = sum(1 for card in game_state.agents[self.id].hand if card is not None and 'One-Eyed Jack' in card)
        num_central_positions = sum(1 for r in range(10) for c in range(10) if (r, c) in self.central_squares and game_state.board.chips[r][c] == self.colour)
        state_vector = np.array([num_legal_actions, num_place_actions, num_remove_actions,
                                 own_seqs, opp_seqs, hand_size, num_jacks, num_two_eyed_jacks, num_one_eyed_jacks, num_central_positions], dtype=np.float32)
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

    def _evaluate_action(self, action, game_state):
        score = 0.0
        opp_id = 1 - self.id
        my_colour = self.colour
        opp_colour = BLU if my_colour == RED else RED

        if action['type'] == 'place':
            r, c = action['coords']
            board = game_state.board.chips

            # Check if this position helps form a sequence (5 in a row/column/diagonal)
            # Check row
            row_count = 0
            has_neighbor = False
            for j in range(max(0, c - 4), min(10, c + 5)):
                if board[r][j] == my_colour:
                    row_count += 1
                    has_neighbor = True
                else:
                    row_count = 0 if board[r][j] != -1 else row_count
                if row_count >= 4:  # 4 chips already, this placement could complete a sequence
                    score += 2000
                    break

            # Check column
            col_count = 0
            for i in range(max(0, r - 4), min(10, r + 5)):
                if board[i][c] == my_colour:
                    col_count += 1
                    has_neighbor = True
                else:
                    col_count = 0 if board[i][c] != -1 else col_count
                if col_count >= 4:
                    score += 2000
                    break

            # Check diagonal (top-left to bottom-right)
            diag1_count = 0
            for d in range(-4, 5):
                i, j = r + d, c + d
                if 0 <= i < 10 and 0 <= j < 10:
                    if board[i][j] == my_colour:
                        diag1_count += 1
                        has_neighbor = True
                    else:
                        diag1_count = 0 if board[i][j] != -1 else diag1_count
                    if diag1_count >= 4:
                        score += 2000
                        break

            # Check diagonal (top-right to bottom-left)
            diag2_count = 0
            for d in range(-4, 5):
                i, j = r + d, c - d
                if 0 <= i < 10 and 0 <= j < 10:
                    if board[i][j] == my_colour:
                        diag2_count += 1
                        has_neighbor = True
                    else:
                        diag2_count = 0 if board[i][j] != -1 else diag2_count
                    if diag2_count >= 4:
                        score += 2000
                        break

            # Boost for central squares
            if (r, c) in self.central_squares:
                score += 500  # Increased from 200 to 500

            # Boost if using a Two-Eyed Jack in a key position
            if 'play_card' in action and action['play_card'] and 'Two-Eyed Jack' in action['play_card']:
                score += 1000  # Increased from 600 to 1000

            # Penalty for isolated placements
            if not has_neighbor:
                score -= 50

        elif action['type'] == 'remove':
            r, c = action['coords']
            board = game_state.board.chips
            # Debug print to inspect values
            print(f"Evaluating remove action at ({r},{c}), board[{r}][{c}]={board[r][c]}, opp_colour={opp_colour}")

            # Check if removing this chip disrupts an opponent's near-sequence
            opp_row_count = 0
            for j in range(max(0, c - 4), min(10, c + 5)):
                if board[r][j] == opp_colour:
                    opp_row_count += 1
                else:
                    opp_row_count = 0 if board[r][j] != -1 else opp_row_count
                if opp_row_count >= 4:
                    score += 3000  # Increased from 1500 to 3000
                    break

            # Check column
            opp_col_count = 0
            for i in range(max(0, r - 4), min(10, r + 5)):
                if board[i][c] == opp_colour:
                    opp_col_count += 1
                else:
                    opp_col_count = 0 if board[i][c] != -1 else opp_col_count
                if opp_col_count >= 4:
                    score += 3000  # Increased from 1500 to 3000
                    break

        # Boost for drafting a Jack
        if 'draft_card' in action and action['draft_card']:
            if 'Two-Eyed Jack' in action['draft_card']:
                score += 300
            elif 'One-Eyed Jack' in action['draft_card']:
                score += 200

        return score

    def SelectAction(self, legal_game_actions, game_state):
        if not legal_game_actions:
            return None

        state_vector = self._convert_state_to_vector(game_state, legal_game_actions)
        state_tensor = torch.from_numpy(state_vector).float().unsqueeze(0)

        self.qnetwork_local.eval()
        with torch.no_grad():
            all_q_values = self.qnetwork_local(state_tensor).squeeze(0)
        self.qnetwork_local.train()

        candidate_actions = []
        for game_action_dict in legal_game_actions:
            action_map_idx = self._get_or_create_action_map_idx(game_action_dict)
            heuristic_score = self._evaluate_action(game_action_dict, game_state)
            if action_map_idx is not None:
                q_value = all_q_values[action_map_idx].item()
                # Combine Q-value and heuristic score (normalize Q-value to a similar scale)
                combined_score = q_value + heuristic_score
                candidate_actions.append((combined_score, game_action_dict, action_map_idx))
            else:
                candidate_actions.append((heuristic_score, game_action_dict, None))

        if random.random() < self.epsilon:
            chosen_game_action = random.choice(legal_game_actions)
        else:
            candidate_actions.sort(key=lambda x: x[0], reverse=True)
            chosen_game_action = candidate_actions[0][1]

        self._current_chosen_action_map_idx = self._get_or_create_action_map_idx(chosen_game_action)
        return chosen_game_action

    def load_model(self, prefix=None):
        if prefix is None:
            prefix = self.DEFAULT_MODEL_LOAD_PREFIX
        try:
            # Define the path to load weights from t_094 directory
            load_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 't_094'))
            local_path = os.path.join(load_dir, f"{prefix}_local.pth")
            target_path = os.path.join(load_dir, f"{prefix}_target.pth")
            print(f"Attempting to load model files: {local_path}, {target_path}")
            self.qnetwork_local.load_state_dict(torch.load(local_path))
            self.qnetwork_target.load_state_dict(torch.load(target_path))
            print(f"Successfully loaded model files: {local_path}, {target_path}")
        except Exception as e:
            print(f"Failed to load model files {prefix}_local.pth or {prefix}_target.pth from {load_dir}. Error: {str(e)}. Using random weights.")

def myAgent(_id):
    return HybridAgent(_id)