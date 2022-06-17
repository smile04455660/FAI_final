
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

        self.conv = nn.Sequential(
            nn.Conv2d(23, 20, 5),
            nn.ReLU(),
            nn.Conv2d(20, 20, 5),
            nn.ReLU(),
            nn.Conv2d(20, 20, 5),
            nn.ReLU(),
            nn.Conv2d(20, 20, 5),
            nn.ReLU(),
            nn.Flatten(0)
        )

        self.action_head = nn.Linear(20, 3)
        self.value_head = nn.Linear(20, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        if len(x.shape) < 4:
            x = x.unsqueeze(0)
        x = self.conv(x)
        action_prob = F.softmax(self.action_head(x), dim=0)
        state_values = self.value_head(x)

        return action_prob, state_values


class MyPlayer(BasePokerPlayer):
    def __init__(self):
        self.fold_ratio = self.call_ratio = raise_ratio = 1.0 / 3

        self.model = RLModel()
        import os
        if os.path.exists("model.ckpt"):
            self.model.load_state_dict(torch.load("model.ckpt"))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)
        self.eps = np.finfo(np.float32).eps.item()
        self.gamma = 0.99


    def declare_action(self, valid_actions, hole_card, round_state):
        state = self._encode_actor_inputs(
            valid_actions, hole_card, round_state)
        probs, state_value = self.model(state)
        m = Categorical(probs)
        action = m.sample()
        self.model.saved_actions.append(
            SavedAction(m.log_prob(action), state_value))
        choice = valid_actions[action]
        action = choice["action"]
        amount = choice["amount"]
        if action == "raise":
            portion = torch.sigmoid(state_value).item()
            mini, maxi = amount["min"], amount["max"]
            amount = portion * (maxi - mini) + mini
        print("===declartation===")
        print(
            f"(probs, value):  ({probs.detach().numpy()}, {state_value.item()})")

        return action, amount

    def _encode_all_cards(self, private_cards, community_cards):
        ret = torch.zeros(private_cards.shape[1:], dtype=int)
        tmp = torch.cat([private_cards.int(), community_cards.int()], dim=0)
        for i in range(len(tmp)):
            ret = torch.bitwise_or(ret, tmp[i])
        return ret.unsqueeze(0)

    def _encode_actor_inputs(self, valid_actions, hole_card, round_state):
        # action: fold, call, raise; 0, 1, 2
        private_cards = self._encode_cards(hole_card, False)
        community_cards = self._encode_cards(
            round_state['community_card'], True)
        all_cards = self._encode_all_cards(private_cards, community_cards)
        street = {"preflop": 4, "flop": 3, "turn": 2,
                  "river": 1}[round_state['street']]
        rounds_left = self._encode_number(4, street)
        bigblind_amt = round_state['small_blind_amount']*2
        pot_size = self._encode_bet(
            bigblind_amt, round_state['pot']['main']['amount'])

        enemy_i, enemy_seat = [(i, seat) for (i, seat) in
                               enumerate(round_state['seats']) if seat['uuid'] != self.uuid][0]
        my_i, my_seat = [(i, seat) for (i, seat) in
                         enumerate(round_state['seats']) if seat['uuid'] == self.uuid][0]
        enemy_stack = self._encode_bet(bigblind_amt, enemy_seat['stack'])
        my_stack = self._encode_bet(bigblind_amt, my_seat['stack'])
        position = self._encode_number(1, my_i)
        enemy_history = self._encode_history(
            bigblind_amt, 4, self.enemy_actions)
        my_history = self._encode_history(bigblind_amt, 3, self.my_actions)
        ret = torch.cat([private_cards, community_cards, all_cards, rounds_left,
                        pot_size, enemy_stack, my_stack, position, enemy_history, my_history], dim=0)
        return ret

    def _encode_history(self, bigblind_amt, number, history):
        ret = []
        for i in range(number):
            amount = history[i]["amount"] if i < len(history) else 0
            ret.append(self._encode_bet(bigblind_amt, amount))
        ret = torch.cat(ret, dim=0)
        return ret

    def _encode_cards(self, cards, isCommunity):
        cards = [self._encode_card(card) for card in cards]
        while isCommunity and len(cards) < 5:
            cards.append(torch.zeros((1, 17, 17)))
        cards = torch.cat(cards, dim=0)
        return cards

    def _decode_round_state(self, round_state):
        pass

    def _encode_number(self, number, value):
        ret = torch.zeros((number, 17, 17))
        ret[:value] = 1
        return ret

    def _encode_bet(self, bigblind_amt, bet):
        # divided by big blind amount
        import math
        num = math.floor(bet / bigblind_amt)
        num = min(num, 52)
        ret = torch.zeros((17, 17))
        for _ in range(num):
            r = num // 4
            s = num % 4
            ret[s][r] = 1

        return ret.unsqueeze(0)

    def _encode_card(self, card):
        suit_map = {'C': 0, 'D': 1, 'H': 2, 'S': 3}
        rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5,
                    '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
        suit = suit_map[card[0]]
        rank = rank_map[card[1]]
        ret = torch.zeros((17, 17))
        ret[suit][rank] = 1

        return ret.unsqueeze(0)

    def receive_game_start_message(self, game_info):
        self.game_initial_stack = game_info['rule']['initial_stack']
        self.my_actions = []
        self.enemy_actions = []

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.round_count = round_count

    def _get_stack_from_seats(self, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        return
        print("===street start===")
        print("street: ", json.dumps(street))
        print("round_state: ", json.dumps(round_state))
        self.street_initial_stack = self._get_stack_from_seats(
            round_state['seats'])

    def receive_game_update_message(self, new_action, round_state):
        self.pot_amount = round_state["pot"]["main"]["amount"]
        print("===game update===")
        # print("new_action: ", json.dumps(new_action))
        # print("round_state: ", json.dumps(round_state))
        if new_action['player_uuid'] == self.uuid:
            reward = -new_action['amount']
            self.model.rewards.append(reward)
            print(f"in update: (a, r): ({self.model.saved_actions},{self.model.rewards})")

    def receive_round_result_message(self, winners, hand_info, round_state):
        print("===round result===")
        # print("winners: ", json.dumps(winners))
        # print("hand_info: ", json.dumps(hand_info))
        # print("round_state: ", json.dumps(round_state))

        # training code
        R = 0
        saved_actions = self.model.saved_actions
        policy_losses = []
        value_losses = []
        returns = []

        if len(self.model.rewards) > 0:
            final_reward = (1 if winners[0]["uuid"]
                            == self.uuid else -1) * self.pot_amount
            print("final_reward", final_reward)
            self.model.rewards[-1] += final_reward

        for r in self.model.rewards[::-1]:
            print('r:', r)
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        # print("ret_b:", returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + self.eps)
        # print("sa:", saved_actions)
        # print("ret:", returns)

            for (log_prob, value), R in zip(saved_actions, returns):
                advantage = R - value.item()
                policy_losses.append(-log_prob * advantage)
                value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

            self.optimizer.zero_grad()
            print("policy_losses:", policy_losses)
            print("value_losses:", value_losses)
            print("rewards:", self.model.rewards)
            loss = torch.stack(policy_losses).sum() + \
                torch.stack(value_losses).sum()
            loss.backward()
            self.optimizer.step()

            if self.round_count % 5 == 4:
                torch.save(self.model.state_dict(), "model.ckpt")

            del self.model.rewards[:]
            del self.model.saved_actions[:]


def setup_ai():
    return MyPlayer()
