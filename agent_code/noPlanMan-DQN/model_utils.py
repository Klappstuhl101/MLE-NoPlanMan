'''
noPlanMan-DQN
Model and memory classes and computation functions to compute features etc.
'''

import math
import os
import queue
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import events as e

'''
HYPERPARAMETERS

SAMPLE_SIZE: sample size of transitions
GAMMA: discount factor

TAU: update rate of target network
LR: learning rate

GRADIENT_CLIP: value of gradient clip 
'''
SAMPLE_SIZE = 512
GAMMA = 0.9

TAU = 0.005
LR = 1e-4

GRADIENT_CLIP = 100


'''
EXP_START: random exploration start value
EXP_END: random exploration end value
EXP_DECAY: decay of random exploration, 4000 yields that model takes over most of the steps after 600-700 rounds
'''
EXP_START = 0.9
EXP_END = 0.05
EXP_DECAY = 4000


'''
LAYER_WIDTH: nodes in each hidden layer
(two hidden layers with 1120 nodes satisfy MaMpf file size constraint)

TRANSITION_HISTORY_SIZE: number of maximal transitions stored
RECORD_ENEMY_TRANSITIONS: probability to record enemy transitions
'''
# LAYER_WIDTH = 7500
LAYER_WIDTH = 1120

TRANSITION_HISTORY_SIZE = 5000
RECORD_ENEMY_TRANSITIONS = 0


'''
Factors that multiply distances, such that coin appears nearer, which gives enemies and crates lower priority
'''
PRIORITY_ENEMY_FACTOR = 2
PRIORITY_CRATE_FACTOR = 2 * PRIORITY_ENEMY_FACTOR


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
DIRECTIONS = {'up': 0, 'down': 1, 'left': 2, 'right': 3, 'None': -1}
PRIORITY_DICT: dict[str, int] = {
    'None': -1, 'coin': 0, 'enemy': 1, 'crate': 2, 'escape': 3, 'dropBomb': 4}

FEATURES = [
    # 'possibleUp',
    # 'possibleDown',
    # 'possibleLeft',
    # 'possibleRight',
    'crateDirection',
    # 'crateUp',
    # 'crateDown',
    # 'crateLeft',
    # 'crateRight',
    # 'coinReachable',
    'coinDirection',
    # 'shortestCoinDistance',
    # 'standingInsideExplosion',
    'escapeDirection',
    # 'shortestEscapeDistance',
    # 'closestBombDuration',
    # 'closestPlayerReachable',
    'playerDirection',
    # 'shortestPlayerDistance',
    # 'bombReady',
    'placeBomb',
    'priority'
]


# Custom Events
MOVE_TO_COIN = "MOVE_TO_COIN"
MOVE_AWAY_FROM_COIN = "MOVE_AWAY_FROM_COIN"
MOVE_TO_CRATE = "MOVE_TO_CRATE"
MOVING_INSIDE_DANGERZONE = "MOVING_INSIDE_DANGERZONE"
MOVING_OUT_OF_DANGERZONE = "MOVING_OUT_OF_DANGERZONE"
LEFT_DANGERZONE = "LEFT_DANGERZONE"
ENTERED_DANGERZONE = "ENTERED_DANGERZONE"
MOVE_TO_ENEMY = "MOVE_TO_ENEMY"
MOVE_AWAY_FROM_ENEMY = "MOVE_AWAY_FROM_ENEMY"
SURVIVED_TICK_NOT_WAITING = "SURVIVED_TICK"
DROPPED_PLAUSIBLE_BOMB = "DROPPED_PLAUSIBLE_BOMB"
DROPPED_IMPLAUSIBLE_BOMB = "DROPPED_IMPLAUSIBLE_BOMB"
FOLLOWED_PRIORITY = "FOLLOWED_PRIORITY"
IGNORED_PRIORITY = "IGNORED_PRIORITY"

MOVE_REWARD: int = 0
INVALID_REWARD: int = -500
PIRORITY_REWARD = 500

REWARDS: dict[str, int] = {
    e.MOVED_UP: MOVE_REWARD,
    e.MOVED_DOWN: MOVE_REWARD,
    e.MOVED_LEFT: MOVE_REWARD,
    e.MOVED_RIGHT: MOVE_REWARD,
    e.WAITED: INVALID_REWARD,
    e.INVALID_ACTION: INVALID_REWARD,
    e.CRATE_DESTROYED: 0,
    e.COIN_COLLECTED: 250,
    e.KILLED_OPPONENT: 300,
    e.OPPONENT_ELIMINATED: 0,
    e.KILLED_SELF: -2500,
    e.GOT_KILLED: -2500,
    e.SURVIVED_ROUND: 10000,
    MOVE_TO_COIN: 50,
    MOVE_AWAY_FROM_COIN: -200,
    MOVE_TO_CRATE: 10,
    MOVING_INSIDE_DANGERZONE: -15000,
    MOVING_OUT_OF_DANGERZONE: 3000,
    LEFT_DANGERZONE: 3000,
    ENTERED_DANGERZONE: 0,
    MOVE_TO_ENEMY: 10,
    MOVE_AWAY_FROM_ENEMY: -11,
    SURVIVED_TICK_NOT_WAITING: 50,
    DROPPED_PLAUSIBLE_BOMB: 1000,
    DROPPED_IMPLAUSIBLE_BOMB: -15000,
    FOLLOWED_PRIORITY: PIRORITY_REWARD,
    IGNORED_PRIORITY: int(-PIRORITY_REWARD * 1.2)
}

'''
Model saving and statistics intervals
'''
# plot every N_ROUNDS (if too small, might increase training duration)
N_ROUNDS = 10
# moving average gets computed over last AVG_STEPS rounds
AVG_STEPS = 50
SAVE_EVERY = 10  # save model every ... round

'''
MODEL NAME and save paths
'''
# model_name = 'NoPlanMan-DQN_coin-agent_v07' # trained on empty

# based on coin-agent
# model_name = 'NoPlanMan-DQN_crate-destroyer_v08'  # trained on crates

# based on crate-destroyer
# trained in arena against peaceful, coin, rule_based_agent
model_name = "NoPlanMan-DQN_killer_v09"


model_path = os.path.join('Models', model_name+'.pt')
avg_reward_path = os.path.join('Models', model_name+'_avg')
memory_path = os.path.join('Memory', model_name+'_memory-deque-pickled')


'''
Transition tuple for memory
'''
Transition = namedtuple(
    'Transition',
    ('state', 'action', 'next_state', 'reward')
)


class Memory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, sample_size):
        transitionSample = random.sample(self.memory, sample_size)
        batch = Transition(*zip(*transitionSample))

        # Filter out end-of-round states
        end_of_round_states = torch.tensor(tuple(
            map(lambda s: s is not None, batch.next_state)), dtype=torch.bool).to(self.device)
        next_step_state_batch = torch.cat(
            [s for s in batch.next_state if s is not None]).to(self.device)

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).unsqueeze(1)
        reward_batch = torch.cat(batch.reward)

        return end_of_round_states, next_step_state_batch, state_batch, action_batch, reward_batch

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, observation_count, action_count):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(observation_count, LAYER_WIDTH)
        self.layer2 = nn.Linear(LAYER_WIDTH, LAYER_WIDTH)
        # self.layer3 = nn.Linear(LAYER_WIDTH, LAYER_WIDTH)
        self.layer4 = nn.Linear(LAYER_WIDTH, LAYER_WIDTH)
        self.layer5 = nn.Linear(LAYER_WIDTH, action_count)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        # x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return self.layer5(x)


def BFS(field, bombs, explosion_map, coins, selfPlayer, otherPlayers):
    """
    Performs BFS to compute shortest paths to various entities and avoids dangers.

    :param field: Game field layout.
    :param bombs: List of bombs with timers.
    :param explosion_map: Map of active explosions.
    :param coins: List of coin positions.
    :param selfPlayer: Current agent's position and state.
    :param otherPlayers: Positions of other agents.

    :return: 
        - crateDirection: int
        - shortestCrateDistance: int
        - coinReachable: bool
        - coinDirection: int
        - shortestCoinDistance: int
        - standingInsideExplosion: bool
        - escapeDirection: int
        - shortestEscapeDistance: int
        - closestBombDuration: int
        - closestPlayerReachable: bool
        - playerDirection: int
        - shortestEnemyDistance: int
        - dropBomb: bool
        - priority: int
    """
    fullField = field.copy()
    selfx, selfy = selfPlayer[3]

    '''
    On fullField:
    - Bombs marked as values 2, 3, 4, 5
    - (future) explosions as 6
    - other players as 7
    - coins as 8
    - potential own bomb explosion as 9
    '''

    for j in range(1, 4):  # 3 explosion fields
        if selfy+j < len(fullField[0]):
            if fullField[selfx][selfy + j] == 0 or fullField[selfx][selfy + j] == 8:
                fullField[selfx][selfy + j] = 9

        if selfy-j > 0:
            if fullField[selfx][selfy - j] == 0 or fullField[selfx][selfy - j] == 8:
                fullField[selfx][selfy - j] = 9

        if selfx + j < len(fullField):
            if fullField[selfx + j][selfy] == 0 or fullField[selfx + j][selfy] == 8:
                fullField[selfx + j][selfy] = 9

        if selfx - j > 0:
            if fullField[selfx - j][selfy] == 0 or fullField[selfx - j][selfy] == 8:
                fullField[selfx - j][selfy] = 9

    for i, player in enumerate(otherPlayers):
        fullField[player[3][0]][player[3][1]] = 7

    for i, coin in enumerate(coins):
        fullField[coin[0]][coin[1]] = 8

    closestBombDistance = float('inf')
    # closestBombFound = False
    closestBombDuration = -1

    for i, bomb in enumerate(bombs):
        bombx, bomby = bomb[0]
        dist = manhattanDistance((selfx, selfy), (bombx, bomby))
        if dist < closestBombDistance:
            closestBombDistance = dist
            closestBombDuration = bomb[1]
        fullField[bombx][bomby] = 2+i
        for j in range(1, 4):  # 3 explosion fields
            if bomby+j < len(fullField[0]):
                if fullField[bombx][bomby + j] == 0 or fullField[bombx][bomby + j] == 8 or fullField[bombx][bomby + j] == 9:
                    fullField[bombx][bomby + j] = 6

            if bomby-j > 0:
                if fullField[bombx][bomby - j] == 0 or fullField[bombx][bomby - j] == 8 or fullField[bombx][bomby - j] == 9:
                    fullField[bombx][bomby - j] = 6

            if bombx + j < len(fullField):
                if fullField[bombx + j][bomby] == 0 or fullField[bombx + j][bomby] == 8 or fullField[bombx + j][bomby] == 9:
                    fullField[bombx + j][bomby] = 6

            if bombx - j > 0:
                if fullField[bombx - j][bomby] == 0 or fullField[bombx - j][bomby] == 8 or fullField[bombx - j][bomby] == 9:
                    fullField[bombx - j][bomby] = 6

    fullField[explosion_map.astype(bool)] = 6

    standingInsideExplosion = (fullField[selfx][selfy] == 6) or (fullField[selfx][selfy] == 2) or (
        fullField[selfx][selfy] == 3) or (fullField[selfx][selfy] == 4) or (fullField[selfx][selfy] == 5)

    coinReachable = False
    closestCoin = (float('inf'), float('inf'))
    closestCrateFound = False
    closestCrate = (float('inf'), float('inf'))
    closestPlayerReachable = False
    closestEnemy = (float('inf'), float('inf'))
    escapedExplosion = False
    closestEscape = (float('inf'), float('inf'))
    potentialEscapeRouteExists = False

    visited = np.zeros(field.shape).astype(bool)
    visited[selfx][selfy] = True
    parent = {(selfx, selfy): (selfx, selfy)}
    q = queue.Queue()
    q.put((selfx, selfy))

    while not q.empty():
        x, y = q.get()

        stepDirections = np.array([(x-1, y), (x+1, y), (x, y+1), (x, y-1)])

        # such that agent not only dodges in the same direction
        # ! does not work for correct reward handout
        # np.random.shuffle(stepDirections)

        for sx, sy in stepDirections:
            if not (sx >= 0 and sx < len(field) and sy >= 0 and sy < len(field[0])):
                continue
            if visited[sx][sy]:
                continue
            else:
                visited[sx][sy] = True
                parent[(sx, sy)] = (x, y)

            match fullField[sx][sy]:
                case 0:
                    # movement possible
                    if standingInsideExplosion and not escapedExplosion:
                        escapedExplosion = True
                        closestEscape = (sx, sy)
                    if not potentialEscapeRouteExists:
                        potentialEscapeRouteExists = True
                        closestPotentialEscape = (sx, sy)
                    q.put((sx, sy))

                case 1:
                    # crate
                    if not closestCrateFound:
                        closestCrateFound = True
                        closestCrate = (sx, sy)

                # case 2:
                #     if not closestBombFound:
                #         closestBombFound = True
                #         closestBombDuration = bombs[0][1]

                # case 3:
                #     if not closestBombFound:
                #         closestBombFound = True
                #         closestBombDuration = bombs[1][1]

                # case 4:
                #     if not closestBombFound:
                #         closestBombFound = True
                #         closestBombDuration = bombs[2][1]

                # case 5:
                #     if not closestBombFound:
                #         closestBombFound = True
                #         closestBombDuration = bombs[3][1]

                case 6:
                    # explosion
                    # only be able to walk inside explosion, if we are already inside of one, no stepping into
                    if standingInsideExplosion and not escapedExplosion:
                        q.put((sx, sy))

                case 7:
                    # enemy
                    if not closestPlayerReachable:
                        closestPlayerReachable = True
                        closestEnemy = (sx, sy)

                case 8:
                    # coin
                    q.put((sx, sy))
                    if not coinReachable:
                        coinReachable = True
                        closestCoin = (sx, sy)

                case 9:
                    # potential explosion, we can walk through freely
                    q.put((sx, sy))
                    if standingInsideExplosion and not escapedExplosion:
                        escapedExplosion = True
                        closestEscape = (sx, sy)

                case _:
                    # no movement possible in this direction
                    pass

        if coinReachable and closestCrateFound and closestPlayerReachable and escapedExplosion and potentialEscapeRouteExists:
            break

    shortestCoinDistance = manhattanDistance((selfx, selfy), closestCoin)
    shortestEnemyDistance = manhattanDistance((selfx, selfy), closestEnemy)
    shortestEscapeDistance = manhattanDistance((selfx, selfy), closestEscape)
    shortestCrateDistance = manhattanDistance((selfx, selfy), closestCrate)

    priority = PRIORITY_DICT['None']
    distances = []
    # ! order according to PRIORITY_DICT
    distances.append(shortestCoinDistance)
    distances.append(shortestEnemyDistance * PRIORITY_ENEMY_FACTOR)
    # * multiply with factor to give lower priority to enemies and crates
    distances.append(shortestCrateDistance * PRIORITY_CRATE_FACTOR)

    if np.min(distances) < float('inf'):
        priority = np.argmin(np.array(distances))

    crateStart = not closestCrateFound
    playerStart = not closestPlayerReachable
    coinStart = not coinReachable
    escapeStart = not escapedExplosion

    #! Only backtrack priority

    while not crateStart or not playerStart or not coinStart or not escapeStart:
        if not crateStart:
            crateParent = parent[(closestCrate)]
            if crateParent == (selfx, selfy):
                crateStart = True
            else:
                closestCrate = crateParent

        if not playerStart:
            playerParent = parent[(closestEnemy)]
            if playerParent == (selfx, selfy):
                playerStart = True
            else:
                closestEnemy = playerParent

        if not coinStart:
            coinParent = parent[(closestCoin)]
            if coinParent == (selfx, selfy):
                coinStart = True
            else:
                closestCoin = coinParent

        if not escapeStart:
            escapeParent = parent[(closestEscape)]
            if escapeParent == (selfx, selfy):
                escapeStart = True
            else:
                closestEscape = escapeParent

    crateDirection = DIRECTIONS['None']
    playerDirection = DIRECTIONS['None']
    coinDirection = DIRECTIONS['None']
    escapeDirection = DIRECTIONS['None']

    if escapedExplosion:
        escapeDirection = determineDirection((selfx, selfy), closestEscape)
    if coinReachable:
        coinDirection = determineDirection((selfx, selfy), closestCoin)
    if closestPlayerReachable:
        playerDirection = determineDirection((selfx, selfy), closestEnemy)
    if closestCrateFound:
        crateDirection = determineDirection((selfx, selfy), closestCrate)

    dropBomb = (shortestEnemyDistance == 1 or shortestCrateDistance ==
                1) and selfPlayer[2] and potentialEscapeRouteExists

    if dropBomb and (shortestCoinDistance > 3):
        priority = PRIORITY_DICT['dropBomb']

    if shortestEscapeDistance != float('inf'):
        priority = PRIORITY_DICT['escape']

    # * Reset directions for all that are not needed, should give more stability in training
    match priority:
        case 0:  # Coin
            crateDirection = DIRECTIONS['None']
            playerDirection = DIRECTIONS['None']
            escapeDirection = DIRECTIONS['None']
        case 1:  # Enemy
            crateDirection = DIRECTIONS['None']
            coinDirection = DIRECTIONS['None']
            escapeDirection = DIRECTIONS['None']
            if shortestEnemyDistance == 1:
                playerDirection = DIRECTIONS['None']
                priority = PRIORITY_DICT['None']
        case 2:  # Crate
            playerDirection = DIRECTIONS['None']
            coinDirection = DIRECTIONS['None']
            escapeDirection = DIRECTIONS['None']
            if shortestCrateDistance == 1:
                crateDirection = DIRECTIONS['None']
                priority = PRIORITY_DICT['None']
        case 3:  # Escape
            playerDirection = DIRECTIONS['None']
            coinDirection = DIRECTIONS['None']
            crateDirection = DIRECTIONS['None']
        case 4:  # Drop Bomb
            escapeDirection = DIRECTIONS['None']
            playerDirection = DIRECTIONS['None']
            coinDirection = DIRECTIONS['None']
            crateDirection = DIRECTIONS['None']
        case -1:
            escapeDirection = DIRECTIONS['None']
            playerDirection = DIRECTIONS['None']
            coinDirection = DIRECTIONS['None']
            crateDirection = DIRECTIONS['None']
            dropBomb = False
        case _:
            pass

    return crateDirection, shortestCrateDistance, coinReachable, coinDirection, shortestCoinDistance, standingInsideExplosion, escapeDirection, shortestEscapeDistance, closestBombDuration, closestPlayerReachable, playerDirection, shortestEnemyDistance, dropBomb, priority


def determineDirection(selfPosition, entityPosition):
    """
    Determines the direction from the agent to the target entity.

    :param selfPosition: Position of the agent.
    :param entityPosition: Position of the target entity.
    :return: Direction toward the entity.
    """
    if selfPosition[0] - entityPosition[0] == 1:
        return DIRECTIONS['left']
    elif selfPosition[0] - entityPosition[0] == -1:
        return DIRECTIONS['right']
    elif selfPosition[1] - entityPosition[1] == 1:
        return DIRECTIONS['up']
    elif selfPosition[1] - entityPosition[1] == -1:
        return DIRECTIONS['down']
    else:
        return DIRECTIONS['None']


def manhattanDistance(position1: tuple[int, int], position2: tuple[int, int]):
    """
    Calculates Manhattan distance between two positions.

    :param position1: First position as (x, y).
    :param position2: Second position as (x, y).
    :return: Manhattan distance between the two positions.
    """
    return abs(position1[0] - position2[0]) + abs(position1[1]-position2[1])


def calcExplorationThreshold(sessionSteps: int):
    """
    Calculates the exploration threshold for epsilon-greedy exploration.

    :param sessionSteps: Current step count in training.
    :return: Exploration threshold based on decay.
    """
    return EXP_END + (EXP_START - EXP_END) * math.exp(-1. * float(sessionSteps) / EXP_DECAY)
