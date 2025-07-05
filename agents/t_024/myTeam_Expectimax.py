import time
import random
from template import Agent
from copy import deepcopy
from Sequence.sequence_utils import RED, BLU, JOKER, EMPTY

# Agent entry point required by the Sequence game framework
def myAgent(_id):
    return Expectimax(_id)

class Expectimax(Agent):
    """
    Expectimax Agent for Sequence game.
    Core features:
    - Depth-limited Expectimax search (max_depth).
    - Follows official APIs only (safe deepcopy and successor generation).
    - Uses robust time management with a safety buffer.
    - Incorporates basic opponent modeling by assuming random or best actions.
    """
    def __init__(self, _id):
        super().__init__(_id)
        self.id = _id
        self.max_depth = 3  # Depth for lookahead.
        self.time_limit = 0.95  # Ensures agent finishes within safe limit (<1s).

    def SelectAction(self, actions, game_state):
        """
        Entry point called every turn.
        Iterates all legal actions.
        Uses Expectimax search to pick the best one under time limits.
        """
        if not actions:
            return None  # No actions, return None defensively.

        start_time = time.time()
        best_action = None
        best_score = -float('inf')

        # Randomize action order to avoid deterministic tie-breaking bias.
        random.shuffle(actions)

        for action in actions:
            if time.time() - start_time > self.time_limit:
                break  # Exit if running out of time.

            successor = self._simulate_action(game_state, action, self.id)
            score = self._expectimax_search(successor, self.max_depth - 1, False, start_time)

            if score > best_score:
                best_score = score
                best_action = action

        # Fallback in rare cases where no action evaluated (due to time).
        if not best_action:
            best_action = random.choice(actions)

        return best_action

    def _expectimax_search(self, state, depth, is_max_node, start_time):
        """
        Recursive Expectimax search.
        Alternates between maximizing node (agent) and averaging node (opponent).
        """
        if time.time() - start_time > self.time_limit or depth == 0 or self._is_terminal(state):
            return self._evaluate_board_state(state)

        agent_to_move = self.id if is_max_node else (1 - self.id)
        actions = self._get_legal_actions(state, agent_to_move)
        if not actions:
            return self._evaluate_board_state(state)

        if is_max_node:
            # Max node: we want the highest expected value.
            return max(self._expectimax_search(self._simulate_action(state, action, agent_to_move), depth - 1, False, start_time) for action in actions)
        else:
            # Expectation node: assume opponent plays randomly or optimally (average).
            values = [self._expectimax_search(self._simulate_action(state, action, agent_to_move), depth - 1, True, start_time) for action in actions]
            return sum(values) / len(values) if values else self._evaluate_board_state(state)

    def _simulate_action(self, state, action, agent_id):
        """
        Generates the next state by applying the given action.
        Uses safe deepcopy and models only visible state changes.
        Does not attempt to model hidden deck or hand replenishment.
        """
        simulated_state = deepcopy(state)
        player_state = simulated_state.agents[agent_id]
        opponent_id = (agent_id + 1) % 2
        opponent_state = simulated_state.agents[opponent_id]

        card = action['play_card']
        if card and card in player_state.hand:
            player_state.hand.remove(card)
            player_state.discard = card

        row, col = action['coords']
        if action['type'] == 'place':
            if simulated_state.board.chips[row][col] in [EMPTY, JOKER]:
                simulated_state.board.chips[row][col] = player_state.colour
        elif action['type'] == 'remove':
            if simulated_state.board.chips[row][col] == opponent_state.colour:
                simulated_state.board.chips[row][col] = EMPTY
        elif action['type'] == 'trade':
            player_state.trade = True

        return simulated_state

    def _get_legal_actions(self, state, agent_id):
        """
        Generates all legal actions for the given player.
        Based purely on current hand and visible board state.
        """
        actions = []
        player_state = state.agents[agent_id]
        opponent_color = state.agents[(agent_id + 1) % 2].colour

        for card in player_state.hand:
            card_lower = card.lower()
            if card_lower in ('jd', 'jc'):
                # Two-eyed Jack can place anywhere.
                for r in range(10):
                    for c in range(10):
                        if state.board.chips[r][c] in [EMPTY, JOKER]:
                            actions.append({'type': 'place', 'play_card': card, 'coords': (r, c)})
            elif card_lower in ('jh', 'js'):
                # One-eyed Jack can remove any opponent chip.
                for r in range(10):
                    for c in range(10):
                        if state.board.chips[r][c] == opponent_color:
                            actions.append({'type': 'remove', 'play_card': card, 'coords': (r, c)})
            else:
                # Regular cards can only place on empty or JOKER.
                for r in range(10):
                    for c in range(10):
                        if state.board.chips[r][c] in [EMPTY, JOKER]:
                            actions.append({'type': 'place', 'play_card': card, 'coords': (r, c)})

        return actions

    def _evaluate_board_state(self, state):
        """
        Heuristic evaluation of the board.
        Key features:
        - Sequence lead (weighted heavily).
        - Center square control.
        - Line potentials (basic).
        - Jack bonuses.
        """
        my_state = state.agents[self.id]
        opp_state = state.agents[(1 - self.id)]

        score = 1000 * (my_state.completed_seqs - opp_state.completed_seqs)

        # Center square heuristic.
        center_positions = [(4, 4), (4, 5), (5, 4), (5, 5)]
        my_center = sum(1 for r, c in center_positions if state.board.chips[r][c] == my_state.colour)
        opp_center = sum(1 for r, c in center_positions if state.board.chips[r][c] == opp_state.colour)
        score += 50 * my_center
        score -= 100 * opp_center

        # Line potential heuristic.
        for r in range(10):
            for c in range(10):
                if state.board.chips[r][c] in [EMPTY, JOKER, my_state.colour]:
                    score += self._line_potential(state, r, c, my_state.colour)
                if state.board.chips[r][c] in [EMPTY, JOKER, opp_state.colour]:
                    score -= 2 * self._line_potential(state, r, c, opp_state.colour)

        # Bonus for Jack possession.
        for card in my_state.hand:
            if card.lower() in ('jd', 'jc'):
                score += 50
            elif card.lower() in ('jh', 'js'):
                score += 30

        return score

    def _line_potential(self, state, r, c, color):
        """
        Simple line potential scorer.
        Checks all 4 directions from (r, c).
        Rewards longer potential lines.
        """
        score = 0
        directions = [((1, 0), (-1, 0)), ((0, 1), (0, -1)),
                      ((1, 1), (-1, -1)), ((1, -1), (-1, 1))]

        for dir_pair in directions:
            length = 1
            for dr, dc in dir_pair:
                for i in range(1, 5):
                    nr, nc = r + dr * i, c + dc * i
                    if 0 <= nr < 10 and 0 <= nc < 10:
                        chip = state.board.chips[nr][nc]
                        if chip == color or chip == JOKER:
                            length += 1
                        else:
                            break
            score += {5: 100, 4: 50, 3: 20, 2: 10}.get(length, 0)

        return score

    def _is_terminal(self, state):
        """
        Checks whether the game has ended due to:
        - Sequence win condition.
        - Center 4-square dominance.
        """
        if any(plr.completed_seqs >= 2 for plr in state.agents):
            return True

        center_positions = [(4, 4), (4, 5), (5, 4), (5, 5)]
        if all(state.board.chips[r][c] == RED for r, c in center_positions):
            return True
        if all(state.board.chips[r][c] == BLU for r, c in center_positions):
            return True

        return False


