from typing import Tuple, NamedTuple
from random import randint, choice
from enum import Enum

State = Tuple[int, int]

class Action(Enum):
    """
    An enum class for action
    """

    stick = 0
    hit = 1

def step(cur_state:State, action:Action) -> Tuple[State, int]:
    """
    Input: current state
    Output: (next_state, reward)
    """

    if action == Action.stick:
        return stick(cur_state)
    else:
        return hit(cur_state)

def init_state() -> State:
    """
    Output: an initial state.
    """
    return (drawBlack(), drawBlack())

def random_state() -> State:
    """
    Output: an random state.
    """
    return (randint(1, 22), drawBlack())


def draw() -> int:
    return randint(1, 10) * choice((1, 1, -1))

def drawBlack() -> int:
    return randint(1, 10)


def in_range(x: int) -> bool:
    return 1 <= x <= 21

def hit(cur_state:State) -> Tuple[State, int]:
    next_state = (cur_state[0] + draw(), cur_state[1])
    if not in_range(next_state[0]):
        return (None, -1)
    else:
        return (next_state, 0)
    
def stick(cur_state:State) -> Tuple[State, int]:
    my, he = cur_state
    
    while he < 17:
        he += draw()
        if not in_range(he):
            he = -1
            break
    

    if my < he:
        return (None, -1)
    elif my == he:
        return (None, 0)
    else:
        return (None, 1)



if __name__ == '__main__':

    W, N = 0., 0

    while N <= 100000:
        s = init_state()

        while s is not None:
            # a = int(input())
            # a = 1 if s[0] < 16 else 0
            a = 0
            if a == 0:
                s, r = step(s, Action.stick)
            else:
                s, r = step(s, Action.hit)

        W += 0.5 * r + 0.5
        N += 1

    print(W/N)

