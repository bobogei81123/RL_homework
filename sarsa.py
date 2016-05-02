from game import State, Action, step, init_state, random_state
from typing import Dict, Tuple, List
from collections import defaultdict as DefaultDict
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
from math import isclose

StateAction = Tuple[State, Action]


N0 = 100
Q_monte = {} # type: Dict[State, float]
with open('Q.mar', 'rb') as f:
    Q_monte = pickle.load(f)

def sarsa(lamb, plot=False):

    random.seed(1)
    value = DefaultDict(float) # type: Dict[StateAction, float]
    N_s = DefaultDict(int) # type: Dict[State, int]
    N_sa = DefaultDict(int) # type: Dict[StateAction, int]
    T = 1000
    if plot:
        X, Y = [], []  # type: List[int], List[float]

    def best_action(state:State) -> Action:
        v_hit, v_stick = value[(state, Action.hit)], value[(state, Action.stick)]
        if v_hit > v_stick:
            return Action.hit
        else:
            return Action.stick

    def choose_action(state:State) -> Action:
        epsilon = N0 / (N0 + N_s[state]) # type: float
        rnd = random.random()
        if rnd < epsilon:
            return random.choice([Action.hit, Action.stick])
        else:
            return best_action(state)


    def run() -> int:

        E = DefaultDict(float)  # type: Dict[StateAction, float]
        state = random_state()
        action = choose_action(state)


        while state is not None:
            next_state, reward = step(state, action)

            if next_state is None:
                next_action = None
                q_next = 0.0
            else:
                next_action = choose_action(next_state)
                q_next = value[(next_state, next_action)]

            delta = reward + q_next - value[(state, action)] 
            N_s[state] += 1
            N_sa[(state, action)] += 1
            E[(state, action)] += 1

            for (s, a) in E:
                alpha = 1.0 / N_sa[(s, a)]
                value[(s, a)] += alpha * E[(s, a)] * delta
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
                    sa = ((i, j), a)
                    err += (value[sa] - Q_monte[sa]) ** 2
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
