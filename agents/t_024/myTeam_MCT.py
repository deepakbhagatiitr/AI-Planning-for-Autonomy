# INFORMATION ------------------------------------------------------------------------------------------------------- #

# IMPORTS AND CONSTANTS ----------------------------------------------------------------------------------------------#

import time
import numpy as np
from copy import deepcopy
import traceback
import logging

from Sequence.sequence_model import COORDS, SequenceState
from Sequence.sequence_utils import BLU, EMPTY, JOKER, RED
from AlteredGameRule_MCT import AlteredGameRule

THINKTIME = 0.9
C_PARAM   = 0.1
EPSILON   = 0.1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    filename="simulation.log",
    filemode="a"
)

# Factory Method used by Sequence Game to create a new Instance of MCSTAgent
def myAgent(_id: int):
    return MCSTAgent(_id)

class MCSTAgent():

    class Node():
        def __init__(self, state: SequenceState, parent_node=None, parent_action=None):
            self.state = deepcopy(state)
            self.parent_node = parent_node
            self.parent_action = parent_action
            self.children_nodes = []
            self.untried_actions = []
            self.N = 0   # visit count
            self.Q = 0   # cumulative reward

        def is_fully_expanded(self):
            return len(self.untried_actions) == 0
        
        def get_best_child_node(self):
            total_N = self.N
            best_score = -float('inf')
            best = None
            for child in self.children_nodes:
                if child.N == 0:
                    score = float('inf')
                else:
                    exploit = child.Q / child.N
                    explore = C_PARAM * np.sqrt(2 * np.log(total_N) / child.N)
                    score = exploit + explore
                if score > best_score:
                    best_score = score
                    best = child
            return best

    def __init__(self, _id: int):
        self.id = _id
        self.game_rule = AlteredGameRule()

    def _is_terminal_state(self, state: SequenceState) -> bool:
        scores = {RED: 0, BLU: 0}
        for plr in state.agents:
            scores[plr.colour] += plr.completed_seqs
        return scores[RED] >= 2 or scores[BLU] >= 2 or len(state.board.draft) == 0

    def _heuristic_score(self, state: SequenceState, action: dict) -> float:
        """
        Heuristic evaluation:
          - Adjacent own chips: +1
          - Adjacent sequence chips: +1
          - Adjacent JOKER: +0.5
          - Adjacent opponent chips: -1
          - Long-range alignment (+3) if same-card spot 5 away with clear line
          - Center-heart positions +2
        """
        if action.get('type') != 'place' or action.get('coords') is None:
            return 0.0

        ps = state.agents[self.id]
        my_col, my_seq = ps.colour, ps.seq_colour
        opp_col, opp_seq = ps.opp_colour, ps.opp_seq_colour
        r, c = action['coords']
        score = 0.0
        # adjacency
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 10 and 0 <= nc < 10:
                chip = state.board.chips[nr][nc]
                if chip == my_col or chip == my_seq:
                    score += 1.0
                elif chip == JOKER:
                    score += 0.5
                elif chip == opp_col or chip == opp_seq:
                    score -= 1.0
        # long-range alignment
        card = action.get('play_card')
        for tr, tc in COORDS[card]:
            if (tr, tc) == (r, c):
                continue
            dr, dc = tr - r, tc - c
            dist = max(abs(dr), abs(dc))
            if dist == 5 and (dr == 0 or dc == 0 or abs(dr) == abs(dc)):
                step_r, step_c = dr//dist, dc//dist
                clear = True
                for i in range(1, dist):
                    rr, cc = r+step_r*i, c+step_c*i
                    chip = state.board.chips[rr][cc]
                    if chip == opp_col or chip == opp_seq:
                        clear = False; break
                if clear:
                    score += 3.0
        # center-heart bonus
        if (r, c) in [(4,4),(4,5),(5,4),(5,5)]:
            score += 2.0
        return score

    def _select(self, node: Node) -> Node:
        while node.is_fully_expanded():
            node = node.get_best_child_node()
        return node

    def _expand(self, node: Node) -> Node:
        if node.untried_actions:
            # pick best untried by heuristic
            idx = max(range(len(node.untried_actions)), key=lambda i: self._heuristic_score(node.state, node.untried_actions[i]))
            action = node.untried_actions.pop(idx)
            next_state = self.DoAction(node.state, action)
            child = self.Node(next_state, node, action)
            child.untried_actions = self.GetActions(next_state)
            node.children_nodes.append(child)
            return child
        return node

    def _simulate(self, node: Node, max_depth: int = 100) -> int:
        sim = deepcopy(node.state)
        depth = 0
        while not self._is_terminal_state(sim) and depth < max_depth:
            actions = self.GetActions(sim)
            if not actions: break
            if np.random.rand() < EPSILON:
                act = np.random.choice(actions)
            else:
                act = max(actions, key=lambda a: self._heuristic_score(sim, a))
            sim = self.DoAction(sim, act)
            depth += 1
        return sim.agents[self.id].completed_seqs - sim.agents[1-self.id].completed_seqs

    def _back_propagate(self, node: Node, reward: int):
        while node:
            node.N += 1
            node.Q += reward
            node = node.parent_node

    def SelectAction(self, legal_actions, root_state):
        # guard
        if not legal_actions or self._is_terminal_state(root_state):
            return legal_actions[0] if legal_actions else None
        # immediate wins
        old = root_state.agents[self.id].completed_seqs
        best_gain, best_act = 0, None
        for act in legal_actions:
            ts = deepcopy(root_state)
            ns = self.DoAction(ts, act)
            gain = ns.agents[self.id].completed_seqs - old
            if gain >= 2: return act
            if gain > best_gain: best_gain, best_act = gain, act
        if best_gain == 1: return best_act
        # MCTS
        root = self.Node(root_state)
        root.untried_actions = legal_actions.copy()
        start = time.time()
        while time.time() < start + THINKTIME:
            leaf = self._select(root)
            if not self._is_terminal_state(leaf.state):
                leaf = self._expand(leaf)
            reward = self._simulate(leaf)
            self._back_propagate(leaf, reward)
        best = max(root.children_nodes, key=lambda n: n.Q/n.N)
        return best.parent_action

    def GetActions(self, state) -> list:
        return self.game_rule.getLegalActions(state, self.id)

    def DoAction(self, state, action) -> SequenceState:
        return self.game_rule.generateSuccessor(state, action, self.id)

# END FILE -----------------------------------------------------------------------------------------------------------#
