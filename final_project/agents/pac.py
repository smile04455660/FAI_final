from calendar import c
from game.players import BasePokerPlayer
import random as rand

import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, history_dim):
        super(Actor, self).__init__()
        pass

class PacPlayer(BasePokerPlayer):
    def __init__(self):
        pass

    def _encode_cards(self, cards):
        t = torch.zeros(4, 13)
        for card in cards:
            rank = {'A':0,'2':1,'3':2,'4':3,'5':4,'6':5,'7':6,'8':7,'9':8,'T':9,'J':10,'Q':11,'K':12}[card[0]]
            suit = {'S':0,'H':1,'D':2,'C':3}[card[1]]
            t[suit][rank] = 1
        return t

    def _encode_card_state(self, hole_card, community_card):
        hole_card_state = self._encode_cards(hole_card)
        n = len(community_card)
        community_card_state = []
        i = 0
        for num in [3, 1, 1]:
            if i  + num>= n: break
            cards = community_card[i: num]
            community_card_state.append(self._encode_cards(cards))
            i += num
        community_card_state = torch.tensor(community_card_state)
        return torch.cat(hole_card_state, community_card_state)



    def declare_action(self, valid_actions, hole_card, round_state):
        card_state = self._encode_card_state(hole_card, round_state['community_card'])
        print(card_state)


        pass

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, new_action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


def setup_ai():
    return AcPlayer()
