from game import State, Action, step, init_state, random_state
from typing import Dict, Tuple, List
from collections import defaultdict as DefaultDict
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle

random.seed(1)

StateAction = Tuple[State, Action]

value = DefaultDict(float) # type: Dict[StateAction, float]
N_s = DefaultDict(int) # type: Dict[State, int]
N_sa = DefaultDict(int) # type: Dict[StateAction, int]
N0 = 100

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

def play(state=None) -> Tuple[List[StateAction], int]:
    if state is None:
        state = init_state()
    r_sum = 0
    history = []  # type: List[StateAction]

    while state is not None:
        action = choose_action(state)
        history.append((state, action))

        state, r = step(state, action)
        r_sum += r

    return (history, r_sum)

def run() -> int:

    history, reward = play(random_state())

    for (state, action) in history:
        N_s[state] += 1
        N_sa[(state, action)] += 1
        alpha = 1.0 / N_sa[(state, action)]

        value[(state, action)] += alpha * (reward - value[(state, action)])


    return reward
        


def main():
    N, R = 0, 0
    for _ in range(1000):
        _, r = play()
        N += 1
        if r > 0:
            R += 1
        elif r == 0:
            R += 0.5
    print('Win rate = %.6f' % (R/N))

    N, R = 0, 0
    for _ in range(1000000):
        r = run()
        N += 1
        if r > 0:
            R += 1

    N, R = 0, 0.
    for _ in range(1000):
        _, r = play()
        N += 1
        R += r * 0.5 + 0.5
    print('Win rate = %.6f' % (R/N))

    X = np.array(range(1, 22))
    Y = np.array(range(1, 11))
    X, Y = np.meshgrid(X, Y)
    data = np.zeros(X.shape)

    for i in range(1, 22):
        for j in range(1, 11):
            action = best_action((i, j))
            val = value[((i, j), action)]
            data[j-1, i-1] = val
            # print('%d' % (1 if action == Action.hit else 0), end='\t')
            # print('%.2f(%d)' % (val, N_sa[((i, j), action)]), end='\t')
        # print()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, data)
    plt.show()
    with open('Q.pickle', 'wb') as out_file:
        pickle.dump(value, out_file)


if __name__ == '__main__':
    main()

