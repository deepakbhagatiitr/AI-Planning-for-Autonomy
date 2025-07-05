import time
from copy import deepcopy
from Sequence.sequence_model import COORDS, SequenceGameRule, SequenceState
from Sequence.sequence_utils import EMPTY, JOKER, BLU, RED
import random
from Sequence.sequence_utils import TRADSEQ, HOTBSEQ, MULTSEQ
class AlteredGameRule(SequenceGameRule):
    def __init__(self, num_of_agent=2):
        super().__init__(num_of_agent)

    def generateSuccessor(self, state, action, agent_id):
        """
        Generate a new state after applying an action, including draft card selection.
        Parameters:
        - state: SequenceState object
        - action: Dict with 'type' ('place', 'remove', 'trade'), 'play_card', 'draft_card', 'coords'
        - agent_id: Integer ID of the agent
        Returns: New SequenceState
        """
        print(f"[Agent {agent_id}] Generating successor for action: {action}")
        start_time = time.time()

        # Deepcopy the entire state
        new_state = deepcopy(state)

        plr_state = new_state.agents[agent_id]
        plr_state.last_action = action
        reward = 0

        # Handle action
        card = action['play_card']
        if card and card in plr_state.hand:
            print(f"[Agent {agent_id}] Removing card {card} from hand")
            plr_state.hand.remove(card)
            plr_state.discard = card

        if action['type'] == 'trade':
            print(f"[Agent {agent_id}] Trading dead card {card}")
            plr_state.trade = True
            draft_card = action.get('draft_card')
            if draft_card and draft_card in new_state.board.draft:
                print(f"[Agent {agent_id}] Selecting draft card {draft_card}")
                new_state.board.draft.remove(draft_card)
                plr_state.hand.append(draft_card)
            print(f"[Agent {agent_id}] Successor generation took {time.time() - start_time:.3f}s")
            return new_state

        r, c = action['coords']
        if action['type'] == 'place':
            if self.is_valid_placement(new_state.board, r, c, plr_state.colour):
                print(f"[Agent {agent_id}] Placing chip at ({r},{c})")
                new_state.board.chips[r][c] = plr_state.colour
                if (r, c) in new_state.board.empty_coords:
                    new_state.board.empty_coords.remove((r, c))
                if plr_state.colour not in new_state.board.plr_coords:
                    new_state.board.plr_coords[plr_state.colour] = []
                new_state.board.plr_coords[plr_state.colour].append((r, c))
                print(f"[Agent {agent_id}] Board state at ({r},{c}) after placement: {new_state.board.chips[r][c]}")
        elif action['type'] == 'remove':
            if self.is_valid_removal(new_state.board, r, c, plr_state.opp_colour):
                print(f"[Agent {agent_id}] Removing opponent chip at ({r},{c})")
                new_state.board.chips[r][c] = EMPTY
                new_state.board.empty_coords.append((r, c))
                if plr_state.opp_colour in new_state.board.plr_coords:
                    if (r, c) in new_state.board.plr_coords[plr_state.opp_colour]:
                        new_state.board.plr_coords[plr_state.opp_colour].remove((r, c))

        # Check for sequence completion
        if action['type'] == 'place':
            seq_info, seq_type = self.checkSeq(new_state.board.chips, plr_state, (r, c))
            if seq_info:
                reward += seq_info['num_seq']
                new_state.board.new_seq = seq_type
                for sequence in seq_info['coords']:
                    for sr, sc in sequence:
                        if new_state.board.chips[sr][sc] != JOKER:
                            new_state.board.chips[sr][sc] = plr_state.seq_colour
                            if (sr, sc) in new_state.board.plr_coords[plr_state.colour]:
                                new_state.board.plr_coords[plr_state.colour].remove((sr, sc))
                plr_state.completed_seqs += seq_info['num_seq']
                plr_state.seq_orientations.extend(seq_info['orientation'])
                print(f"[Agent {agent_id}] Completed {seq_info['num_seq']} sequences. Total: {plr_state.completed_seqs}")

        plr_state.trade = False
        plr_state.score += reward

        draft_card = action.get('draft_card')
        if draft_card and draft_card in new_state.board.draft:
            print(f"[Agent {agent_id}] Selecting draft card {draft_card}")
            new_state.board.draft.remove(draft_card)
            plr_state.hand.append(draft_card)

        print(f"[Agent {agent_id}] Successor generation took {time.time() - start_time:.3f}s")
        return new_state

    def getLegalActions(self, state, agent_id):
        """
        Generate legal actions for the agent.
        Returns: List of action dictionaries
        """
        start_time = time.time()
        actions = []
        plr_state = state.agents[agent_id]

        draft_cards = state.board.draft[:5] if len(state.board.draft) >= 5 else state.board.draft
        if not draft_cards:
            print(f"[Agent {agent_id}] Warning: No draft cards available!")
            draft_cards = [None]

        for card in plr_state.hand:
            if self.is_dead_card(state.board, card):
                for draft_card in draft_cards:
                    actions.append({'type': 'trade', 'play_card': card, 'draft_card': draft_card, 'coords': None})
                continue

            if 'Jack' not in card:
                for r, c in COORDS[card]:
                    if self.is_valid_placement(state.board, r, c, plr_state.colour):
                        for draft_card in draft_cards:
                            actions.append({'type': 'place', 'play_card': card, 'draft_card': draft_card, 'coords': (r, c)})
            elif 'Two' in card:
                for r, c in state.board.empty_coords:
                    for draft_card in draft_cards:
                        actions.append({'type': 'place', 'play_card': card, 'draft_card': draft_card, 'coords': (r, c)})
            elif 'One' in card:
                for r, c in state.board.plr_coords.get(plr_state.opp_colour, []):
                    if not self.is_part_of_sequence(state.board.chips, r, c):
                        for draft_card in draft_cards:
                            actions.append({'type': 'remove', 'play_card': card, 'draft_card': draft_card, 'coords': (r, c)})

        print(f"[Agent {agent_id}] Generated {len(actions)} legal actions in {time.time() - start_time:.3f}s")
        return actions

    def is_valid_placement(self, board, r, c, colour):
        return (0 <= r < 10 and 0 <= c < 10 and
                board.chips[r][c] == EMPTY and
                (r, c) in board.empty_coords)

    def is_valid_removal(self, board, r, c, opp_colour):
        return (0 <= r < 10 and 0 <= c < 10 and
                board.chips[r][c] == opp_colour and
                not self.is_part_of_sequence(board.chips, r, c))

    def is_dead_card(self, board, card):
        if 'Jack' in card:
            return False
        return all(board.chips[r][c] != EMPTY for r, c in COORDS[card])

    def is_part_of_sequence(self, chips, r, c):
        for dr, dc in [(0,1), (1,0), (1,1), (-1,1)]:
            for start in [-4, 0]:
                valid = True
                coords = []
                for i in range(5):
                    nr, nc = r + (start + i) * dr, c + (start + i) * dc
                    if not (0 <= nr < 10 and 0 <= nc < 10):
                        valid = False
                        break
                    coords.append((nr, nc))
                    if chips[nr][nc] == EMPTY:
                        valid = False
                        break
                if valid and all(chips[nr][nc] != JOKER for nr, nc in coords):
                    return True
        return False

    def checkSeq(self, chips, plr_state, last_coords):
        """
        Check for sequences formed by placing a chip at last_coords.
        Returns: (dict with num_seq, orientation, coords, or None), seq_type
        """
        clr, sclr = plr_state.colour, plr_state.seq_colour
        oc, os = plr_state.opp_colour, plr_state.opp_seq_colour
        seq_type = TRADSEQ
        seq_coords = []
        seq_found = {'vr': 0, 'hz': 0, 'd1': 0, 'd2': 0, 'hb': 0}
        found = False
        nine_chip = lambda x, clr: len(x) == 9 and len(set(x)) == 1 and clr in x
        lr, lc = last_coords

        for r, c in COORDS['jk']:
            chips[r][c] = clr

        coord_list = [(4,4), (4,5), (5,4), (5,5)]
        heart_chips = [chips[y][x] for x, y in coord_list]
        if EMPTY not in heart_chips and (clr in heart_chips or sclr in heart_chips) and not (oc in heart_chips or os in heart_chips):
            seq_type = HOTBSEQ
            seq_found['hb'] += 2
            seq_coords.append(coord_list)
            print(f"[Agent {plr_state.id}] Central squares controlled! Immediate win.")

        # Search for sequences
        vr = [(-4,0), (-3,0), (-2,0), (-1,0), (0,0), (1,0), (2,0), (3,0), (4,0)]
        hz = [(0,-4), (0,-3), (0,-2), (0,-1), (0,0), (0,1), (0,2), (0,3), (0,4)]
        d1 = [(-4,-4), (-3,-3), (-2,-2), (-1,-1), (0,0), (1,1), (2,2), (3,3), (4,4)]
        d2 = [(-4,4), (-3,3), (-2,2), (-1,1), (0,0), (1,-1), (2,-2), (3,-3), (4,-4)]
        for seq, seq_name in [(vr, 'vr'), (hz, 'hz'), (d1, 'd1'), (d2, 'd2')]:
            coord_list = [(r + lr, c + lc) for r, c in seq]
            coord_list = [i for i in coord_list if 0 <= min(i) and 9 >= max(i)]
            chip_str = ''.join([str(chips[r][c]) for r, c in coord_list])
            if nine_chip(chip_str, clr):
                seq_found[seq_name] += 2
                seq_coords.append(coord_list)
                print(f"[Agent {plr_state.id}] Found 9-chip sequence ({seq_name}): {coord_list}")
            if sclr not in chip_str:
                sequence_len = 0
                start_idx = 0
                for i in range(len(chip_str)):
                    if chip_str[i] == str(clr):
                        sequence_len += 1
                    else:
                        start_idx = i + 1
                        sequence_len = 0
                    if sequence_len >= 5:
                        seq_found[seq_name] += 1
                        seq_coords.append(coord_list[start_idx:start_idx + 5])
                        print(f"[Agent {plr_state.id}] Found 5-chip sequence ({seq_name}): {coord_list[start_idx:start_idx + 5]}")
                        break
            else:
                for pattern in [str(clr)*5, str(clr)*4 + str(sclr), str(clr)*3 + str(sclr) + str(clr), str(clr)*2 + str(sclr) + str(clr)*2, str(clr) + str(sclr) + str(clr)*3, str(sclr) + str(clr)*4]:
                    for start_idx in range(5):
                        if chip_str[start_idx:start_idx + 5] == pattern:
                            seq_found[seq_name] += 1
                            seq_coords.append(coord_list[start_idx:start_idx + 5])
                            print(f"[Agent {plr_state.id}] Found 5-chip sequence with existing sequence chip ({seq_name}): {coord_list[start_idx:start_idx + 5]}")
                            found = True
                            break
                    if found:
                        break

        for r, c in COORDS['jk']:
            chips[r][c] = JOKER

        num_seq = sum(seq_found.values())
        if num_seq > 1 and seq_type != HOTBSEQ:
            seq_type = MULTSEQ
        return ({'num_seq': num_seq, 'orientation': [k for k, v in seq_found.items() if v], 'coords': seq_coords}, seq_type) if num_seq else (None, None)