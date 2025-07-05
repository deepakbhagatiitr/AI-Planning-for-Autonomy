import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import time
from collections import deque, namedtuple
from copy import deepcopy
import os
from template import Agent
from Sequence.sequence_model import SequenceGameRule, SequenceState
from Sequence.sequence_utils import EMPTY, JOKER, RED, BLU, RED_SEQ, BLU_SEQ
from AlteredGameRule import AlteredGameRule

# --- DQN Components ---

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


class myAgent(Agent):
    BOARD_ROWS = 10
    BOARD_COLS = 10
    BOARD_CELL_STATES = 6

    ALL_RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
    ALL_SUITS = ['S', 'H', 'D', 'C']
    CARD_STRINGS = []
    for _ in range(2):  # Two decks
        for suit in ALL_SUITS:
            for rank in ALL_RANKS:
                CARD_STRINGS.append(rank + suit)

    CARD_TO_INT_MAP = {card_str: i for i, card_str in enumerate(sorted(list(set(CARD_STRINGS))))}
    VOCAB_SIZE = len(CARD_TO_INT_MAP)
    MAX_HAND_SIZE = 7
    ACTION_SPACE_SIZE_FOR_NETWORK = 1000
    DEFAULT_MODEL_LOAD_PREFIX = "hybrid_sequence_agent"
    THINKTIME = 0.95  # Time limit for Expectimax search

    def __init__(self, _id, num_agents=2, seed=0, training_mode=True, model_load_prefix=None,
                 buffer_size=int(1e4), batch_size=64, gamma=0.99, tau=1e-3, lr=5e-4, update_every=4):
        super().__init__(_id)
        self.id = _id
        self.num_agents = num_agents
        self.training_mode = True 

        # DQN State Representation
        board_flat_size = self.BOARD_ROWS * self.BOARD_COLS * self.BOARD_CELL_STATES
        hand_vector_size = self.VOCAB_SIZE
        other_features_size = 3 
        self.state_size = board_flat_size + hand_vector_size + other_features_size

        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # DQN Components
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
            self.epsilon = 0.05
            self.epsilon_decay = 1.0
            self.epsilon_min = 0.05

        self._current_chosen_action_map_idx = None

        # Expectimax Components
        self.game_rule = AlteredGameRule(num_agents)
        self.max_depth = 1
        self.start_time = None
        self.central_squares = [(4,4), (4,5), (5,4), (5,5)]

        load_prefix = model_load_prefix if model_load_prefix is not None else self.DEFAULT_MODEL_LOAD_PREFIX
        self.load_model(load_prefix)

    def _convert_state_to_vector(self, game_state):
        board_features = np.zeros((self.BOARD_ROWS, self.BOARD_COLS, self.BOARD_CELL_STATES), dtype=np.float32)
        my_agent_state = game_state.agents[self.id]
        opp_agent_id = (self.id + 1) % self.num_agents
        opp_agent_state = game_state.agents[opp_agent_id]

        my_colour_chip = RED if my_agent_state.colour == RED else BLU
        opp_colour_chip = BLU if my_colour_chip == RED else RED
        my_seq_chip_type = RED_SEQ if my_colour_chip == RED else BLU_SEQ
        opp_seq_chip_type = BLU_SEQ if my_colour_chip == RED else RED_SEQ

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
                elif chip_on_board == JOKER:
                    board_features[r, c, 5] = 1.0
        board_flat = board_features.flatten()

        hand_vector = np.zeros(self.VOCAB_SIZE, dtype=np.float32)
        for card_str in my_agent_state.hand:
            if card_str in self.CARD_TO_INT_MAP:
                hand_vector[self.CARD_TO_INT_MAP[card_str]] += 1.0

        my_score_norm = my_agent_state.score / 2.0
        opp_score_norm = opp_agent_state.score / 2.0
        cards_in_hand_norm = len(my_agent_state.hand) / self.MAX_HAND_SIZE if self.MAX_HAND_SIZE > 0 else 0.0

        state_vector = np.concatenate((board_flat, hand_vector,
                                       [my_score_norm, opp_score_norm, cards_in_hand_norm])).astype(np.float32)

        if state_vector.shape[0] != self.state_size:
            raise ValueError(f"State vector length mismatch: expected {self.state_size}, got {state_vector.shape[0]}.")
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
        if not self.training_mode:
            return
        if action_map_idx is None:
            return

        self.memory.add(state_vector, action_map_idx, reward, next_state_vector, done)
        print(f"Added experience, memory size: {len(self.memory)}")  # Debug print to confirm training

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def SelectAction(self, legal_game_actions, game_state):
        if not legal_game_actions:
            return None

        state_vector = self._convert_state_to_vector(game_state)
        state_tensor = torch.from_numpy(state_vector).float().unsqueeze(0)

        self.qnetwork_local.eval()
        with torch.no_grad():
            all_q_values = self.qnetwork_local(state_tensor).squeeze(0)
        if self.training_mode:
            self.qnetwork_local.train()

        candidate_actions = []
        for game_action_dict in legal_game_actions:
            action_map_idx = self._get_or_create_action_map_idx(game_action_dict)
            if action_map_idx is not None:
                q_value = all_q_values[action_map_idx].item()
                candidate_actions.append((q_value, game_action_dict, action_map_idx))
            else:
                candidate_actions.append((-float('inf'), game_action_dict, None))

        if random.random() < self.epsilon:
            chosen_game_action = random.choice(legal_game_actions)
            self._current_chosen_action_map_idx = self._get_or_create_action_map_idx(chosen_game_action)
            if self.training_mode:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            return chosen_game_action

        candidate_actions.sort(key=lambda x: x[0], reverse=True)
        top_k = min(3, len(candidate_actions))
        shortlisted_actions = [action for _, action, _ in candidate_actions[:top_k]]
        shortlisted_action_indices = [idx for _, _, idx in candidate_actions[:top_k]]

        self.start_time = time.time()
        best_action, best_score = None, -float('inf')
        for i, action in enumerate(shortlisted_actions):
            if self.is_time_exceeded():
                break
            next_state = self.game_rule.generateSuccessor(deepcopy(game_state), action, self.id)
            _, score = self.expectimax(next_state, 1 - self.id, depth=1, alpha=-float('inf'), beta=float('inf'))
            if score > best_score:
                best_score = score
                best_action = action
                self._current_chosen_action_map_idx = shortlisted_action_indices[i]

        if best_action is None or self.is_time_exceeded():
            
            best_action = shortlisted_actions[0]
            self._current_chosen_action_map_idx = shortlisted_action_indices[0]

        if self.training_mode:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return best_action

    # --- Expectimax Components ---

    def _is_terminal_state(self, state: SequenceState) -> bool:
        scores = {RED: 0, BLU: 0}
        for plr in state.agents:
            scores[plr.colour] += plr.completed_seqs
        return scores[RED] >= 2 or scores[BLU] >= 2 or len(state.board.draft) == 0

    def check_central_squares(self, state: SequenceState) -> bool:
        ps = state.agents[self.id]
        clr, sclr = ps.colour, ps.seq_colour
        opp_clr, opp_sclr = ps.opp_colour, ps.opp_seq_colour
        heart_chips = [state.board.chips[y][x] for x, y in self.central_squares]
        return (EMPTY not in heart_chips and
                (clr in heart_chips or sclr in heart_chips) and
                not (opp_clr in heart_chips or opp_sclr in heart_chips))

    def _heuristic_score(self, state: SequenceState, action: dict = None) -> float:
        ps = state.agents[self.id]
        opp_id = 1 - self.id
        score = 0.0

        # Sequence completion
        own_seqs = state.agents[self.id].completed_seqs
        opp_seqs = state.agents[opp_id].completed_seqs
        score += 15000 * own_seqs - 10000 * opp_seqs

        # Near-sequence detection
        own_near_4 = self.count_near_sequences(state.board.chips, ps.colour, length=4)
        own_near_3 = self.count_near_sequences(state.board.chips, ps.colour, length=3)
        opp_near_4 = self.count_near_sequences(state.board.chips, state.agents[opp_id].colour, length=4)
        opp_near_3 = self.count_near_sequences(state.board.chips, state.agents[opp_id].colour, length=3)
        score += 2000 * own_near_4 + 500 * own_near_3 - 3000 * opp_near_4 - 1000 * opp_near_3

        # Central squares
        central_control = sum(1 for r, c in self.central_squares if state.board.chips[r][c] == ps.colour)
        score += 10000 * central_control

        # Evaluate action
        if action:
            if action['type'] == 'place':
                r, c = action['coords']
                temp_chips = [row[:] for row in state.board.chips]
                temp_chips[r][c] = ps.colour
                new_own_near_4 = self.count_near_sequences(temp_chips, ps.colour, length=4)
                new_own_near_3 = self.count_near_sequences(temp_chips, ps.colour, length=3)
                if new_own_near_4 > own_near_4:
                    score += 2000
                if new_own_near_3 > own_near_3:
                    score += 500
                if (r, c) in self.central_squares:
                    score += 10000
            elif action['type'] == 'remove':
                r, c = action['coords']
                temp_chips = [row[:] for row in state.board.chips]
                temp_chips[r][c] = EMPTY
                new_opp_near_4 = self.count_near_sequences(temp_chips, state.agents[opp_id].colour, length=4)
                if new_opp_near_4 < opp_near_4:
                    score += 5000
            if 'draft_card' in action:
                draft_card = action['draft_card']
                if 'Two' in draft_card:
                    score += 20
                elif 'One' in draft_card:
                    score += 10
        return score

    def count_near_sequences(self, board, colour, length):
        count = 0
        directions = [(0,1), (1,0), (1,1), (-1,1)]
        for r in range(10):
            for c in range(10):
                for dr, dc in directions:
                    chip_count = 0
                    for i in range(length):
                        nr, nc = r + i*dr, c + i*dc
                        if not (0 <= nr < 10 and 0 <= nc < 10):
                            break
                        chip = board[nr][nc]
                        if chip == colour or chip == JOKER or ((nr, nc) in [(0,0), (0,9), (9,0), (9,9)] and chip == EMPTY):
                            chip_count += 1
                    if chip_count >= length:
                        count += 1
                        break
        return count

    def expectimax(self, state, agent_id, depth, alpha, beta):
        if self.is_time_exceeded():
            return None, -float('inf')
        if depth >= self.max_depth or self._is_terminal_state(state):
            return None, self._heuristic_score(state)

        actions = self.game_rule.getLegalActions(state, agent_id)
        if not actions:
            return None, self._heuristic_score(state)

        if agent_id == self.id: 
            best_score = -float('inf')
            best_action = None
            for action in actions:
                if self.is_time_exceeded():
                    break
                next_state = self.game_rule.generateSuccessor(deepcopy(state), action, agent_id)
                _, score = self.expectimax(next_state, 1 - agent_id, depth + 1, alpha, beta)
                if score > best_score:
                    best_score = score
                    best_action = action
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            return best_action, best_score
        else:  
            total_score = 0
            count = 0
            for action in actions[:1]:  # Limit to 1 for efficiency
                if self.is_time_exceeded():
                    break
                next_state = self.game_rule.generateSuccessor(deepcopy(state), action, agent_id)
                _, score = self.expectimax(next_state, self.id, depth + 1, alpha, beta)
                total_score += score
                count += 1
                beta = min(beta, total_score / max(1, count))
                if beta <= alpha:
                    break
            avg_score = total_score / max(1, count) if count > 0 else self._heuristic_score(state)
            return None, avg_score

    def is_time_exceeded(self):
        return (time.time() - self.start_time) >= self.THINKTIME

    # --- DQN Training Methods ---

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

    def save_model(self, filename_prefix="hybrid_sequence_agent"):
        try:
            path_local = f"{filename_prefix}_local.pth"
            torch.save(self.qnetwork_local.state_dict(), path_local)
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, filename_prefix="hybrid_sequence_agent"):
        try:
            path_local = f"{filename_prefix}_local.pth"
            if os.path.exists(path_local):
                self.qnetwork_local.load_state_dict(torch.load(path_local))
                self.qnetwork_target.load_state_dict(torch.load(path_local))
                self.qnetwork_local.train() if self.training_mode else self.qnetwork_local.eval()
                self.qnetwork_target.eval()
            else:
                pass
        except Exception as e:
            print(f"Error loading model: {e}. Starting with new model.")