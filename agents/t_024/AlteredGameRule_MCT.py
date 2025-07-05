
from copy import deepcopy
from Sequence.sequence_model import COORDS, SequenceGameRule
from Sequence.sequence_utils import EMPTY, JOKER


# This class is a modified version of the SequenceGameRule class.
# It overrides generateSuccessor methods to handle the hidden deck problem.
class AlteredGameRule(SequenceGameRule):
    def __init__(self, num_of_agent=2):
        super().__init__(num_of_agent)

    def generateSuccessor(self, state, action, agent_id):
        """
        Modified version that handles the hidden deck
        This simplifies the successor generation by not handling card drawing
        """
        state = deepcopy(state)
        plr_state = state.agents[agent_id]
        plr_state.last_action = action
        reward = 0
        
        # Handle the action without card drawing
        card = action['play_card']
        if card:
            if card in plr_state.hand:
                plr_state.hand.remove(card)
                plr_state.discard = card
                
                # We don't modify the draft or draw new cards, since that's random
        
        # If action was to trade in a dead card, action is complete
        if action['type'] == 'trade':
            plr_state.trade = True
            return state
        
        # Update board state for place/remove actions
        r, c = action['coords']
        if action['type'] == 'place':
            state.board.chips[r][c] = plr_state.colour
            if (r, c) in state.board.empty_coords:
                state.board.empty_coords.remove((r, c))
            if plr_state.colour in state.board.plr_coords:
                state.board.plr_coords[plr_state.colour].append((r, c))
        elif action['type'] == 'remove':
            state.board.chips[r][c] = EMPTY
            state.board.empty_coords.append((r, c))
            if plr_state.opp_colour in state.board.plr_coords:
                if (r, c) in state.board.plr_coords[plr_state.opp_colour]:
                    state.board.plr_coords[plr_state.opp_colour].remove((r, c))
        
        # Check for sequence completion
        if action['type'] == 'place':
            seq, seq_type = self.checkSeq(state.board.chips, plr_state, (r, c))
            if seq:
                reward += seq['num_seq']
                state.board.new_seq = seq_type
                for sequence in seq['coords']:
                    for r, c in sequence:
                        if state.board.chips[r][c] != JOKER:
                            state.board.chips[r][c] = plr_state.seq_colour
                            try:
                                if plr_state.colour in state.board.plr_coords:
                                    state.board.plr_coords[plr_state.colour].remove(action['coords'])
                            except:
                                pass
                plr_state.completed_seqs += seq['num_seq']
                plr_state.seq_orientations.extend(seq['orientation'])
        
        plr_state.trade = False
        plr_state.score += reward

        return state
    