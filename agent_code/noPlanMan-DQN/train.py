
import csv
import os
import pickle
import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import events as e

from . import model_utils as modu
from .callbacks import state_to_features


def setup_training(self):
    """
    Initializes the agent for training.

    :param self: The agent object.
    """

    self.device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    # self.device = "cpu"

    self.policyNet.to(self.device)
    self.targetNet.to(self.device)

    self.optimizer = optim.AdamW(
        self.policyNet.parameters(), lr=modu.LR, amsgrad=True)

    self.lossCriterion = nn.SmoothL1Loss()

    self.memory = modu.Memory(modu.TRANSITION_HISTORY_SIZE)

    try:  # try loading Memory
        with open(modu.memory_path, 'rb') as file:
            self.memory = pickle.load(file)
        self.logger.debug(f'Loaded {len(self.memory)} Memory entries')
        self.actSteps = len(self.memory) // 100
    except:
        self.logger.debug(f'Initializing empty Memory')

    self.rewardSumOfRound = 0.
    self.roundRewards = []
    self.roundSteps = []

    # Initialize history attributes for tracking statistics
    self.reward_history = []
    self.avg_reward_history = []
    self.steps_survived_history = []
    self.coins_collected_history = []
    self.enemies_killed_history = []
    self.crates_destroyed_history = []

    # Initialize counters for current round actions
    self.coins_collected = 0
    self.enemies_killed = 0
    self.crates_destroyed = 0

    return


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Updates memory and rewards based on game events.

    :param self: The agent object.
    :param old_game_state: The state before the action.
    :param self_action: The action taken by the agent.
    :param new_game_state: The state after the action.
    :param events: List of game events.
    """
    

    detected_events = check_events(self,
                                   old_game_state,
                                   self_action,
                                   new_game_state,
                                   events)

    self.logger.info(
        f'Action {self_action} encountered game event(s) {", ".join(map(repr, detected_events))} in step {new_game_state["step"]}')

    stepReward = reward_from_events(self, detected_events)
    self.rewardSumOfRound += stepReward

    # Gather actions for plotting
    self.coins_collected += events.count(e.COIN_COLLECTED)
    self.enemies_killed += events.count(e.KILLED_OPPONENT)
    self.crates_destroyed += events.count(e.CRATE_DESTROYED)

    self.memory.push(
        state_to_features(self, old_game_state),
        torch.tensor([modu.ACTIONS.index(self_action)], device=self.device),
        state_to_features(self, new_game_state),
        torch.tensor([stepReward], device=self.device)
    )

    optimize_model(self)

    # Soft update of state dict
    targetNet_state_dict = self.targetNet.state_dict()
    policyNet_state_dict = self.policyNet.state_dict()
    for key in policyNet_state_dict:
        targetNet_state_dict[key] = policyNet_state_dict[key] * \
            modu.TAU + targetNet_state_dict[key]*(1-modu.TAU)
    self.targetNet.load_state_dict(targetNet_state_dict)

    return


def enemy_game_events_occurred(self, enemy_name: str, old_enemy_game_state: dict, enemy_action: str, new_enemy_game_state: dict, enemy_events: List[str]):
    """
    Records and updates enemy events randomly.

    :param enemy_name: Name of the enemy agent.
    :param old_enemy_game_state: The state before the enemy's action.
    :param enemy_action: The action taken by the enemy.
    :param new_enemy_game_state: The state after the enemy's action.
    :param enemy_events: List of enemy events.
    """

    if random.random() < modu.RECORD_ENEMY_TRANSITIONS:  # randomly save transitions of enemies

        detected_enemy_events = check_events(self,
                                             old_enemy_game_state,
                                             enemy_action,
                                             new_enemy_game_state,
                                             enemy_events)

        self.logger.info(f'RECORDING ACTION of {enemy_name}.')
        self.logger.info(
            f'{enemy_name} encountered game event(s) {", ".join(map(repr, detected_enemy_events))} in step {new_enemy_game_state["step"]}')

        enemyReward = reward_from_events(self, detected_enemy_events)

        if enemy_action is None:
            enemy_action = 'WAIT'

        self.memory.push(
            state_to_features(self, old_enemy_game_state),
            torch.tensor([modu.ACTIONS.index(enemy_action)],
                         device=self.device),
            state_to_features(self, new_enemy_game_state),
            torch.tensor([enemyReward], device=self.device)
        )

    return


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Finalizes rewards, saves and plots data at the end of a round.
    Replaces game_events_occured in the final round.

    :param self: The agent object.
    :param last_game_state: The final state of the game.
    :param last_action: The agent's last action.
    :param events: List of final events.
    """
    self.logger.info(
        f"Final action {last_action} in ending of round {last_game_state['round']}")
    self.logger.info(
        f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    stepReward = reward_from_events(self, events)
    self.rewardSumOfRound += stepReward

    print(f'\n Rewards this round: {self.rewardSumOfRound}')
    survivedSteps = last_game_state['step']
    self.roundSteps.append(survivedSteps)
    print(f'Steps survived this round: {survivedSteps}')

    self.roundRewards.append(self.rewardSumOfRound)
    self.rewardSumOfRound = 0

    # Append these values to their respective histories
    self.coins_collected_history.append(self.coins_collected)
    self.enemies_killed_history.append(self.enemies_killed)
    self.crates_destroyed_history.append(self.crates_destroyed)
    self.reward_history.append(self.roundRewards[-1])
    self.avg_reward_history.append(np.mean(self.roundRewards))
    self.steps_survived_history.append(survivedSteps)

    # reset counter
    self.coins_collected = 0
    self.enemies_killed = 0
    self.crates_destroyed = 0

    self.memory.push(
        state_to_features(self, last_game_state),
        torch.tensor([modu.ACTIONS.index(last_action)], device=self.device),
        None,
        torch.tensor([stepReward], device=self.device)
    )

    optimize_model(self)

    self.logger.info(
        f"############################## ROUND {last_game_state['round']} ENDED ##############################")

    # only save model and Replaymemory after every SAVE_EVERY rounds
    if last_game_state['round'] % modu.SAVE_EVERY == 0:

        # Store ReplayMemory
        with open(modu.memory_path, 'wb') as file:
            pickle.dump(self.memory, file)

        # Store the model, only if performance is better
        # latestMean = np.mean(np.array(self.roundRewards)[-modu.SAVE_EVERY:])

        # model that survives longer
        # latestMean = np.mean(np.array(self.roundSteps)[-modu.SAVE_EVERY:])

        # currentBest = float('-inf')
        # try:
        #     with open(modu.avg_reward_path, 'rb') as file:
        #         currentBest = float(file.read().strip())
        # except:
        #     pass

        torch.save(self.targetNet, modu.model_path)
        # with open(modu.avg_reward_path, 'wb+') as file:
        #     file.write(str(latestMean).encode('utf-8'))

        # if latestMean >= currentBest:
        #     torch.save(self.targetNet, modu.model_path)
        #     with open(modu.avg_reward_path, 'wb+') as file:
        #         file.write(str(latestMean).encode('utf-8'))

    # Plot rewards and steps survived at the end of the last round
    if last_game_state['round'] % modu.N_ROUNDS == 0:
        plot_training(self.reward_history, self.avg_reward_history, self.steps_survived_history,
                      self.coins_collected_history, self.enemies_killed_history, self.crates_destroyed_history)

        save_training_data_to_csv(
            self, self.reward_history, self.avg_reward_history, self.steps_survived_history,
            self.coins_collected_history, self.enemies_killed_history, self.crates_destroyed_history
        )
    return


def reward_from_events(self, events: List[str]) -> int:
    """
    Calculates the total reward from events per round.

    :param self: The agent object.
    :param events: List of game events.
    :return: Total reward.
    """

    reward_sum: int = 0
    for event in events:
        if event in modu.REWARDS:
            reward_sum += modu.REWARDS[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def check_events(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]) -> List[str]:
    """
    Detects and adds specific events based on the game state per round.

    :param self: The agent object.
    :param old_game_state: The state before the action.
    :param self_action: The action taken by the agent.
    :param new_game_state: The state after the action.
    :param events: List of game events.
    :return: Updated list of events.
    """

    # Extract agent's previous and current positions
    old_position = old_game_state['self'][3]
    new_position = new_game_state['self'][3]

    oldCrateDirection, oldShortestCrateDistance, \
        _, oldCoinDirection, oldShortestCoinDistance, \
        oldStandingInsideExplosion, oldEscapeDirection, oldShortestEscapeDistance, \
        _, _, oldEnemyDirection, oldShortestEnemyDistance, \
        oldDropBomb, oldPriority = modu.BFS(
            old_game_state['field'],
            old_game_state['bombs'],
            old_game_state['explosion_map'],
            old_game_state['coins'],
            old_game_state['self'],
            old_game_state['others']
        )

    # Use BFS to get directional data
    newCrateDirection, newShortestCrateDistance, \
        _, newCoinDirection, newShortestCoinDistance, \
        newStandingInsideExplosion, newEscapeDirection, newShortestEscapeDistance, \
        _, _, newEnemyDirection, newShortestEnemyDistance, \
        newDropBomb, newPriority = modu.BFS(
            new_game_state['field'],
            new_game_state['bombs'],
            new_game_state['explosion_map'],
            new_game_state['coins'],
            new_game_state['self'],
            new_game_state['others']
        )

    # Determine the direction moved
    actual_direction = modu.determineDirection(old_position, new_position)

    detected_events = []

    if old_position != new_position:

        # Check if the move was towards or away from a coin
        if e.COIN_COLLECTED not in events:
            if oldCoinDirection != modu.DIRECTIONS['None']:
                # if newShortestCoinDistance < oldShortestCoinDistance:
                if actual_direction == oldCoinDirection:
                    detected_events.append(modu.MOVE_TO_COIN)
                else:
                    detected_events.append(modu.MOVE_AWAY_FROM_COIN)

        # Check if the move was towards or away from a crate
        if oldCrateDirection != modu.DIRECTIONS['None']:
            # if newShortestCrateDistance < oldShortestCrateDistance:
            if actual_direction == oldCrateDirection:
                detected_events.append(modu.MOVE_TO_CRATE)

        # Check if the move was towards or away from an enemy
        if oldEnemyDirection != modu.DIRECTIONS['None']:
            if actual_direction == oldEnemyDirection:
                detected_events.append(modu.MOVE_TO_ENEMY)
            else:
                detected_events.append(modu.MOVE_AWAY_FROM_ENEMY)

    # Check if the agent moved into or out of a danger zone
    if e.BOMB_DROPPED not in events:
        if not oldStandingInsideExplosion and newStandingInsideExplosion:
            detected_events.append(modu.ENTERED_DANGERZONE)

        if e.GOT_KILLED not in events:
            if oldStandingInsideExplosion and not newStandingInsideExplosion:
                detected_events.append(modu.LEFT_DANGERZONE)

        if (oldEscapeDirection != modu.DIRECTIONS['None']) and (modu.ENTERED_DANGERZONE not in detected_events) and (modu.LEFT_DANGERZONE not in detected_events):
            if actual_direction == oldEscapeDirection:
                detected_events.append(modu.MOVING_OUT_OF_DANGERZONE)
            else:
                detected_events.append(modu.MOVING_INSIDE_DANGERZONE)

        if oldPriority == 4:
            events.append(modu.IGNORED_PRIORITY)
    else:
        if oldPriority == 4:
            events.append(modu.DROPPED_PLAUSIBLE_BOMB)
            events.append(modu.FOLLOWED_PRIORITY)
        else:
            events.append(modu.DROPPED_IMPLAUSIBLE_BOMB)
            # commented out, such that IGNORED_PRIORITY is not added twice
            # events.append(modu.IGNORED_PRIORITY)

    if (e.WAITED not in events) and (e.INVALID_ACTION not in events) and (e.GOT_KILLED not in events):
        detected_events.append(modu.SURVIVED_TICK_NOT_WAITING)

    events.extend(detected_events)

    match oldPriority:
        case 0:  # coin
            if modu.MOVE_TO_COIN in events or e.COIN_COLLECTED in events:
                events.append(modu.FOLLOWED_PRIORITY)
            else:
                events.append(modu.IGNORED_PRIORITY)
        case 1:  # enemy
            if modu.MOVE_TO_ENEMY in events:
                events.append(modu.FOLLOWED_PRIORITY)
            else:
                events.append(modu.IGNORED_PRIORITY)
        case 2:  # crate
            if modu.MOVE_TO_CRATE in events:
                events.append(modu.FOLLOWED_PRIORITY)
            else:
                events.append(modu.IGNORED_PRIORITY)
        case 3:  # escape explosion
            if modu.MOVING_OUT_OF_DANGERZONE in events or modu.LEFT_DANGERZONE in events:
                events.append(modu.FOLLOWED_PRIORITY)
            else:
                events.append(modu.IGNORED_PRIORITY)
        case 4:  # drop a bomb
            pass  # handled earlier, since e.BOMB_DROPPED needs to be checked
        case -1:  # waiting
            if e.WAITED in events:
                events.append(modu.FOLLOWED_PRIORITY)
            else:
                events.append(modu.IGNORED_PRIORITY)
        case _:
            pass

    return events


def optimize_model(self):
    """
    Optimizes the model using memory samples.

    :param self: The agent object.
    """

    if len(self.memory) < modu.SAMPLE_SIZE:
        return

    end_of_round_states, next_step_state_batch, state_batch, action_batch, reward_batch = self.memory.sample(
        modu.SAMPLE_SIZE)

    # Compute actions that would have been taken by the policyNet
    policy_actions = self.policyNet(state_batch)
    state_action_values = torch.gather(policy_actions, 1, action_batch)

    # Compute expected state action values for all next states, that are not end of round states
    # computed on the target net
    next_state_values = torch.zeros(modu.SAMPLE_SIZE, device=self.device)
    with torch.no_grad():
        next_state_values[end_of_round_states] = self.targetNet(
            next_step_state_batch).max(1).values

    # Q-values computed
    expected_state_action_values = (
        next_state_values * modu.GAMMA) + reward_batch

    # Loss
    # loss = F.huber_loss(expected_state_action_values.unsqueeze(1), state_action_values)
    loss = self.lossCriterion(state_action_values,
                              expected_state_action_values.unsqueeze(1))

    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    # Gradient clipping
    torch.nn.utils.clip_grad_value_(
        self.policyNet.parameters(), modu.GRADIENT_CLIP)
    self.optimizer.step()

    return


def plot_training(reward_history, avg_reward_history, steps_survived_history, coins_collected_history, enemies_killed_history, crates_destroyed_history):
    """
    Generates and saves training plots.

    :param reward_history: List of rewards.
    :param avg_reward_history: List of average rewards.
    :param steps_survived_history: List of steps survived.
    :param coins_collected_history: List of coins collected.
    :param enemies_killed_history: List of enemies killed.
    :param crates_destroyed_history: List of crates destroyed.
    """
    # Create folder if it doesn't exist
    plots_dir = os.path.join("Plots", modu.model_name)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Rewards and Steps Survived 
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    # First subplot: Rewards and Average Rewards
    axs[0].plot(reward_history, label='Rewards per Round', color='blue')
    axs[0].plot(avg_reward_history, label='Average Reward', color='red')

    # moving_avg gets computed and plottet
    if len(reward_history) >= modu.AVG_STEPS:
        moving_avg = [np.mean(reward_history[i:i + modu.AVG_STEPS])
                      for i in range(len(reward_history) - modu.AVG_STEPS + 1)]
        axs[0].plot(range(modu.AVG_STEPS - 1, len(reward_history)), moving_avg,
                    label=f'Moving Avg (last {modu.AVG_STEPS} rounds)', color='green')

    axs[0].set_title('Rewards and Average Rewards per Round')
    axs[0].set_xlabel('Round')
    axs[0].set_ylabel('Rewards')
    axs[0].legend()

    # Second subplot: Steps Survived
    axs[1].plot(steps_survived_history, label='Steps Survived', color='black')
    axs[1].set_title('Steps Survived per Round')
    axs[1].set_xlabel('Round')
    axs[1].set_ylabel('Steps')
    axs[1].legend()

    plt.tight_layout()

    statistics_filename = os.path.join(
        plots_dir, f"{modu.model_name}_statistics.png")
    plt.savefig(statistics_filename)
    plt.close(fig)

    # Actions (Coins, Enemies, Crates) 
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot coins collected, enemies killed, and crates destroyed
    ax.plot(crates_destroyed_history, 'd',
            label='Crates Destroyed', color='brown')
    ax.plot(coins_collected_history, 'o',
            label='Coins Collected', color='gold', )
    ax.plot(enemies_killed_history, 'x', label='Enemies Killed', color='red')

    ax.set_title('Actions per Round (Coins, Enemies, Crates)')
    ax.set_xlabel('Round')
    ax.set_ylabel('Count')
    ax.legend()

    plt.tight_layout()

    # Save the actions plot
    actions_filename = os.path.join(
        plots_dir, f"{modu.model_name}_actions.png")
    plt.savefig(actions_filename)
    plt.close(fig)

    return


def save_training_data_to_csv(self, reward_history, avg_reward_history, steps_survived_history, coins_collected_history, enemies_killed_history, crates_destroyed_history):
    """
    Saves training data to a CSV file.

    :param self: The agent object.
    :param reward_history: List of rewards.
    :param avg_reward_history: List of average rewards.
    :param steps_survived_history: List of steps survived.
    :param coins_collected_history: List of coins collected.
    :param enemies_killed_history: List of enemies killed.
    :param crates_destroyed_history: List of crates destroyed.
    """
    
    # Define CSV file path 
    csv_file = os.path.join("Plots", modu.model_name,
                            f"{modu.model_name}_training_data.csv")

    
    headers = ['Round', 'Reward', 'Average Reward', 'Steps Survived',
               'Coins Collected', 'Enemies Killed', 'Crates Destroyed']

    # Combine all data into rows for CSV
    data = zip(
        range(1, len(reward_history) + 1),
        reward_history, [round(avg)
                         for avg in avg_reward_history], steps_survived_history,
        coins_collected_history, enemies_killed_history, crates_destroyed_history
    )


    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(headers)
        writer.writerows(data)

        # Append the raw arrays at the end
        writer.writerow([])
        writer.writerow(['Raw Arrays'])
        writer.writerow(['Reward History'] + reward_history)
        writer.writerow(['Average Reward History'] + avg_reward_history)
        writer.writerow(['Steps Survived History'] + steps_survived_history)
        writer.writerow(['Coins Collected History'] + coins_collected_history)
        writer.writerow(['Enemies Killed History'] + enemies_killed_history)
        writer.writerow(['Crates Destroyed History'] +
                        crates_destroyed_history)

    return
