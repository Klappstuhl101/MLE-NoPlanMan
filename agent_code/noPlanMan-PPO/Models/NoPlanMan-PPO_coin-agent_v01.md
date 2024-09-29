# NoPlanMan-PPO_coin-agent_v01

Is not converging really, at least performs bad in training configuration.
May perform okay in testing with non-training act configuration.


BATCH_SIZE = 128               # update policy for K epochs in one PPO update

EPS_CLIP = 0.2          # clip parameter for PPO
GAMMA = 0.99            # discount factor

LR_ACTOR = 0.0003       # learning rate for actor network
LR_CRITIC = 0.001       # learning rate for critic network


# Hyper parameters
TRANSITION_HISTORY_SIZE = 5000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...


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
MOVE_INTO_DANGERZONE = "MOVE_INTO_DANGERZONE"
MOVE_OUT_OF_DANGERZONE = "MOVE_OUT_OF_DANGERZONE"
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
INVALID_REWARD: int = -100

REWARDS: dict[str, int] = {
    e.MOVED_UP: MOVE_REWARD,
    e.MOVED_DOWN: MOVE_REWARD,
    e.MOVED_LEFT: MOVE_REWARD,
    e.MOVED_RIGHT: MOVE_REWARD,
    e.WAITED: INVALID_REWARD,
    e.INVALID_ACTION: INVALID_REWARD,
    e.CRATE_DESTROYED: 50,
    e.COIN_COLLECTED: 250,
    e.KILLED_OPPONENT: 300,
    e.OPPONENT_ELIMINATED: 0,
    e.KILLED_SELF: -5000,
    e.GOT_KILLED: -5000,
    e.SURVIVED_ROUND: 1000,
    MOVE_TO_COIN: 50,
    MOVE_AWAY_FROM_COIN: -65,
    MOVE_TO_CRATE: 10,
    MOVE_INTO_DANGERZONE: -100,
    MOVE_OUT_OF_DANGERZONE: 50,
    LEFT_DANGERZONE: 150,
    ENTERED_DANGERZONE: -200,
    MOVE_TO_ENEMY: 10,
    MOVE_AWAY_FROM_ENEMY: -11,
    SURVIVED_TICK_NOT_WAITING: 50,
    DROPPED_PLAUSIBLE_BOMB: 100,
    DROPPED_IMPLAUSIBLE_BOMB: -100,
    FOLLOWED_PRIORITY: 50,
    IGNORED_PRIORITY: -50
}

Neural net with two hidden layers with 128 neurons.