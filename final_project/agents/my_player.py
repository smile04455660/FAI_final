import json
import numpy as np

from game.players import BasePokerPlayer
import random as rand

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


from collections import namedtuple
import json

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class RLModel(nn.Module):
    def __init__(self):
        super(RLModel, self).__init__()
        self.affine = nn.Linear(23, 128)
        self.action_head = nn.Linear(128, 3)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []


    def forward(self, x):
        x = F.relu(self.affine(x))
        action_prob = F.softmax(self.action_head(x), dim=0)
        state_values = self.value_head(x)

        return action_prob, state_values


class MyPlayer(BasePokerPlayer):
    def __init__(self):
        self.fold_ratio = self.call_ratio = raise_ratio = 1.0 / 3

        self.model = RLModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-2)
        self.eps = np.finfo(np.float32).eps.item()

    def set_action_ratio(self, fold_ratio, call_ratio, raise_ratio):
        ratio = [fold_ratio, call_ratio, raise_ratio]
        scaled_ratio = [1.0 * num / sum(ratio) for num in ratio]
        self.fold_ratio, self.call_ratio, self.raise_ratio = scaled_ratio

    def _encode_actor_inputs(self, valid_actions, hole_card, round_state):
        # action: fold, call, raise; 0, 1, 2
        if len(valid_actions) < 3:
            mini, maxi = 0, 0
        else:
            mini, maxi = valid_actions[2]['amount']['min'], valid_actions[2]['amount']['max']
        hole_card = tuple([self._decode_card(card) for card in hole_card])
        inputs = []
        for ele in hole_card + self._decode_round_state(round_state) + (mini, maxi):
            inputs.append(torch.tensor(ele).flatten())
        inputs = torch.cat(inputs)
        return inputs.float()

    def _decode_round_state(self, round_state):
        street_map = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}

        # real value; (one hot encoding?)
        street = street_map[round_state['street']]
        pot_amount = round_state['pot']['main']['amount']
        # community_card up to five: five slots
        community_card = [self._decode_card(card)
                          for card in round_state['community_card']]
        while len(community_card) < 5:
            community_card.append((0, 0))

        dealer_btn = round_state['dealer_btn']
        pos = 0 if round_state['seats'][0]['uuid'] == self.uuid else 1
        is_big_blind = 0 if round_state['big_blind_pos'] == pos else 0
        small_blind_amount = round_state['small_blind_amount']
        round_state['seats'].sort(key=lambda x: x['uuid'] != self.uuid)
        stacks = [seat['stack'] for seat in round_state['seats']]

        return community_card, street, pot_amount,  dealer_btn, is_big_blind, small_blind_amount, stacks

    def _decode_card(self, card):
        suit_map = {'C': 1, 'D': 2, 'H': 3, 'S': 4}
        rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                    '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        suit = suit_map[card[0]]
        rank = rank_map[card[1]]
        return suit, rank

    def declare_action(self, valid_actions, hole_card, round_state):
        state = self._encode_actor_inputs(valid_actions, hole_card, round_state)
        probs, state_value = self.model(state)
        m = Categorical(probs)
        action = m.sample()
        self.model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        choice = valid_actions[action]
        action = choice["action"]
        amount = choice["amount"]
        if action == "raise":
            portion = torch.sigmoid(state_value).item()
            mini, maxi = amount["min"], amount["max"]
            amount = portion * (maxi - mini) + mini
        
        return action, amount


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
    return MyPlayer()
