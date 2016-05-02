from game import State, Action, step, init_state, random_state
from typing import Dict, Tuple, List
from collections import defaultdict as DefaultDict
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
from math import isclose

StateAction = Tuple[State, Action]

Dealer_range = [(1, 4), (4, 7), (7, 10)]
Player_range = [(1, 6), (4, 9), (7, 12), (10, 15), (13, 18), (16, 21)]



N0 = 100
Q_monte = {} # type: Dict[State, float]
with open('Q.mar', 'rb') as f:
    Q_monte = pickle.load(f)

def sarsa(lamb, plot=False):

    random.seed(1)
    theta = np.zeros(dtype=float, shape=(3, 6, 2))
    N_s = DefaultDict(int) # type: Dict[State, int]
    N_sa = DefaultDict(int) # type: Dict[StateAction, int]
    T = 1000
    if plot:
        X, Y = [], []  # type: List[int], List[float]

    def get_feature(state:State, action:Action):
        w = np.zeros(dtype=float, shape=(3, 6, 2))
        pla, dea = state
        for i, dr in enumerate(Dealer_range):
            if not dr[0] <= dea <= dr[1]:
                continue
            for j, pl in enumerate(Player_range):
                if not pl[0] <= pla <= pl[1]:
                    continue

                a = 1 if action == Action.hit else 0
                w[i, j, a] = 1.
        return w

    def get_value(state:State, action:Action):
        return np.sum(theta * get_feature(state, action))

    def best_action(state:State) -> Action:
        v_hit, v_stick = get_value(state, Action.hit), get_value(state, Action.stick)
        if v_hit > v_stick:
            return Action.hit
        else:
            return Action.stick

    def choose_action(state:State) -> Action:
        epsilon = 0.05
        rnd = random.random()
        if rnd < epsilon:
            return random.choice([Action.hit, Action.stick])
        else:
            return best_action(state)


    def run() -> int:

        E = DefaultDict(float)  # type: Dict[StateAction, float]
        state = random_state()
        action = choose_action(state)
        nonlocal theta

        while state is not None:
            next_state, reward = step(state, action)

            if next_state is None:
                next_action = None
                q_next = 0.0
            else:
                next_action = choose_action(next_state)
                q_next = get_value(next_state, next_action)

            delta = reward + q_next - get_value(state, action) 

            N_s[state] += 1
            N_sa[(state, action)] += 1
            E[(state, action)] += 1

            for (s, a) in E:
                alpha = 0.01
                theta += alpha * E[(s, a)] * delta * get_feature(s, a)
                E[(s, a)] *= lamb

            (state, action) = (next_state, next_action)

        if plot:
            X.append(X[-1]+1 if X else 1)
            Y.append(calc_err())


        return reward

    def calc_err():
        err = 0.
        for i in range(1, 22):
            for j in range(1, 11):
                for a in (Action.hit, Action.stick):
                    s, a = ((i, j), a)
                    err += (get_value(s, a) - Q_monte[(s, a)]) ** 2
                # print('%.2f' % (get_value(s, best_action(s))), end='\t')
            # print()

        return err

    for _ in range(T):
        run()

    err = calc_err()
    Q = {}  # type: Dict[State, float]

    if plot:
        plt.plot(X, Y)
        plt.show()

    return err

def main():
    X, Y = [], []  # type: List[float], List[float]
    for lamb in np.arange(0.0, 1.1, 0.1):
        err = sarsa(lamb, plot=(isclose(lamb, 0.0) or isclose(lamb, 1.0)))
        X.append(lamb)
        Y.append(err)
    plt.plot(X, Y)
    plt.show()

    


if __name__ == '__main__':
    main()
