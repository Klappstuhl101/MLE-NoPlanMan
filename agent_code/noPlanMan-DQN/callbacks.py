
import random

import numpy as np
import torch

from . import model_utils as modu


def setup(self):
    """
    Initializes the agent.

    :param self: The agent object.
    """

    self.device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    # self.device = "cpu"

    self.logger.debug(f"Device {self.device} chosen")

    try:
        print(f"Loading from {modu.model_path}")
        self.logger.debug(f"Loading from {modu.model_path}")
        self.policyNet = torch.load(modu.model_path, map_location=self.device)
        self.targetNet = torch.load(modu.model_path, map_location=self.device)
        self.logger.debug(f'Successfully loaded model')
        print(f"Successfully loaded {modu.model_path}")

    except:
        self.policyNet = modu.DQN(len(modu.FEATURES), len(modu.ACTIONS))
        self.targetNet = modu.DQN(len(modu.FEATURES), len(modu.ACTIONS))
        self.targetNet.load_state_dict(self.policyNet.state_dict())
        self.logger.debug(f'Failed to load model, created new one')
        print(f"Failed to load model")

    self.policyNet.to(self.device)
    self.targetNet.to(self.device)

    self.actSteps = 0


def act(self, game_state: dict) -> str:
    """
    Decides the next action for the agent based on the current game state.

    :param self: The agent object.
    :param game_state: The dictionary describing the current game state.
    :return: str (action to take).
    """

    features = state_to_features(self, game_state)
    # features.to(self.device)

    explorationThreshold = modu.calcExplorationThreshold(self.actSteps)

    if self.train and random.random() < explorationThreshold:
        # choose random action
        self.logger.info("Choosing action purely at random.")
        #     # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(modu.ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    else:
        self.actSteps += 1
        actionIndex = self.policyNet(features).max(1).indices.view(1, 1).item()
        self.logger.info(
            f"Querying model for action. Chose {modu.ACTIONS[actionIndex]}, Index {actionIndex}")
        return modu.ACTIONS[actionIndex]


def state_to_features(self, game_state: dict) -> np.array:
    """
    Converts the game state to a feature vector for the model.

    :param self: The agent object.
    :param game_state: The dictionary describing the current game state.
    :return: torch.tensor (feature vector for the model).
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    round = game_state['round']
    step = game_state['step']
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    coins = game_state['coins']
    selfPlayer = game_state['self']
    otherPlayers = game_state['others']

    crateDirection, shortestCrateDistance, \
        coinReachable, coinDirection, shortestCoinDistance, \
        standingInsideExplosion, escapeDirection, shortestEscapeDistance, closestBombDuration, \
        closestPlayerReachable, enemyDirection, shortestEnemyDistance, \
        dropBomb, priority = modu.BFS(
            field, bombs, explosion_map, coins, selfPlayer, otherPlayers)

    # possibleUp, possibleDown, possibleLeft, possibleRight = modu.getPossibleMoves(
    #     field, bombs, explosion_map, coins, selfPlayer, otherPlayers)

    # bombPositions = [bomb[0] for bomb in bombs]
    # otherPositions = [player[3] for player in otherPlayers]

    # possibleUp = modu.possibleUp(
    #     selfPlayer[3], field, bombPositions, otherPositions)
    # possibleDown = modu.possibleDown(
    #     selfPlayer[3], field, bombPositions, otherPositions)
    # possibleLeft = modu.possibleLeft(
    #     selfPlayer[3], field, bombPositions, otherPositions)
    # possibleRight = modu.possibleRight(
    #     selfPlayer[3], field, bombPositions, otherPositions)

    # crateUp = modu.crateAbove(selfPlayer[3], field)
    # crateDown = modu.crateBelow(selfPlayer[3], field)
    # crateLeft = modu.crateLeft(selfPlayer[3], field)
    # crateRight = modu.crateRight(selfPlayer[3], field)

    # bombReady = selfPlayer[2]
    # selfScore = selfPlayer[1]

    featureVec = [
        # possibleUp,
        # possibleDown,
        # possibleLeft,
        # possibleRight,
        crateDirection,
        # crateUp,
        # crateDown,
        # crateLeft,
        # crateRight,
        # coinReachable,
        coinDirection,
        #   shortestCoinDistance,
        # standingInsideExplosion,
        escapeDirection,
        #   shortestEscapeDistance,
        #   closestBombDuration,
        #   closestPlayerReachable,
        enemyDirection,
        #   shortestPlayerDistance,
        # bombReady,
        dropBomb,
        priority
        #   selfScore
    ]

    return torch.tensor(featureVec, dtype=torch.float, device=self.device).unsqueeze(0)
