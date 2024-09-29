# Parameters for NoPlanMan-DQN_coin-agent_v01

- BATCH_SIZE = 512
- GAMMA = 0.99
- EPS_START = 0.9
- EPS_END = 0.05
- EPS_DECAY = 1000
- TAU = 0.005
- LR = 1e-4


## Hyper parameters
- TRANSITION_HISTORY_SIZE = 5000  # keep only ... last transitions
- RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...


## Rewards

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
    MOVING_INTO_DANGERZONE: -100,
    MOVING_OUT_OF_DANGERZONE: 50,
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