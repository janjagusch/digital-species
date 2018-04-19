import datetime
import gym
import neat
import numpy as np
import os.path
import math
import visualize
import pickle
from tqdm import tqdm

# Flag if time penalty should be applied.
TIME_PENALTY = False
# Flag if lives left reward should be applied.
LIVES_LEFT_REWARD = False
# Flag if movement penalty should be applied.
MOVEMENT_PENALTY = False

# Returns the current date and time.
now = datetime.datetime.now()

# Defines if video should be recorded. Is adjusted inside the program. Do not change.
record_video = False

# Defines gym environment and adds monitor.
env = gym.make('Breakout-ram-v4')
# Creates folder based on current date and time.
folder_name = 'results' + '__' + now.strftime("%Y_%m_%d__%H_%M_%S")
# Adds monitor.
env = gym.wrappers.Monitor(env, folder_name, video_callable=lambda episode_id: record_video, force=True)

# Converts internal actions to emulator actions. Avoid pressing 'fire', as it has no purpose.
actions = {
    0: 0,
    1: 2,
    2: 3
}


def reward_lives_left(lives_left):
    """Returns rewards for number of lives left after beating the game."""

    return lives_left + 1


def standardize_input(input):
    """Standardizes input vector with Z-Score normalization."""

    return (input - np.mean(input)) / np.std(input)


def penalize_time(action_log):
    """Returns penalization factor for time taken."""

    return 1 / math.log(len(action_log))


def penalize_movement(action_log):
    """Penalizes high energy consumption. Analogy to nature: Moving costs energy, whereas staying still does not.
    Calculates relative share of 'noop' actions and uses square root as penalty factor.
    """

    # Remove fire actions from action log.
    actions_filtered = [a for a in action_log if a != 1]
    # Count number of 'noop' actions and calculate 'noop' rate.
    noop_actions = [a for a in actions_filtered if a == 0]
    noop_rate = len(noop_actions) / len(actions_filtered)
    # Calculate penalty factor as square root of ratio.
    penalty = math.sqrt(noop_rate)
    return penalty


def simulate(net, log=False):
    """Simulate breakout game.
    Simulator automatically executes 'fire' action when ball is out of bounds. This speeds up learning process.
    If log, returns final score, action log and lives left. Else, returns final score.
    """

    observation = env.reset()
    # List of executed actions.
    action_log = []
    # Number of lives before episode.
    lives_old = None
    # Number of lives after episode.
    lives_new = None
    # Episode.
    j = 0
    # Total final score.
    total_score = 0.0

    while True:
        # If game has started or life has been list, execute 'fire'.
        if j == 0 or (lives_old != lives_new and lives_old):
            action = 1
        # Else calculate action.
        else:
            output = net.activate(standardize_input(observation))
            action = convert_action(np.argmax(output))
        action_log.append(action)
        observation, reward, done, info = env.step(action)
        total_score += reward
        # Update lives.
        lives_old, lives_new = lives_new, info['ale.lives']
        if done:
            break
        j += 1
    # If log is true, additionally return action list.
    if log:
        return total_score, action_log, lives_new
    else:
        return total_score


def convert_action(action_value):
    """Converts internal action to emulator action."""

    return actions[action_value]


def evaluate_genomes(genomes, config, n_runs=10, time_penalty=TIME_PENALTY,
                     lives_left_reward=LIVES_LEFT_REWARD, movement_penalty=MOVEMENT_PENALTY):
    """Evaluates genomes in population."""

    # Create networks.
    nets = []
    for gid, g in genomes:
        nets.append((g, neat.nn.RecurrentNetwork.create(g, config)))
        g.fitness = []
    # Evaluate networks.
    for genome, net in tqdm(nets):
        fitness = []
        # Takes average of defined number of runs.
        for i in range(n_runs):
            total_score, action_log, lives_left = simulate(net, log=True)
            # Rewards lives left, if applicable.
            if lives_left_reward:
                total_score = total_score * reward_lives_left(lives_left)
            # Penalizes time, if applicable.
            if time_penalty:
                total_score = total_score * penalize_time(action_log)
            # Penalizes movement, if applicable.
            if movement_penalty:
                total_score = total_score * penalize_movement(action_log)
            fitness.append(total_score)
        # Calculate fitness as mean of all runs.
        genome.fitness = np.mean(fitness)


def run(population):
    """Run evolutionary process, based on provided population."""

    # Creates reporter.
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.StdOutReporter(True))
    # Checkpoint every 10 generations or 900 seconds.
    population.add_reporter(neat.Checkpointer(10, 900))
    # Reference to global variable.
    global record_video
    # Run until user interrupts the process.
    while True:
        try:
            # Evolve population for one generation.
            champ = population.run(evaluate_genomes, 1)
            # Save fitness plot in result folder.
            visualize.plot_stats(stats, ylog=False, view=False, filename=os.path.join(folder_name, 'fitness.svg'))
            # Print mean fitness.
            mfs = sum(stats.get_fitness_mean()[-5:]) / 5.0
            print("Average mean fitness over last 5 generations: {0}".format(mfs))
            # Create champion network.
            champ_net = neat.nn.RecurrentNetwork.create(champ, population.config)
            # Allow video recording.
            record_video = True
            # Run champion network.
            score, action_log, lives_left = simulate(champ_net, True)
            # Block video recording.
            record_video = False
            # Print score of champion network.
            print('Score of champion: {}'.format(score))
            # Check if agent completed the game.
            win = True if lives_left > 0 else False
            if win:
                print('Solved. Agent has beaten the game.')
                # Serialize champion network.
                file_name = 'champion' + '__' + champ.key + '.pkl'
                with open(os.path.join(folder_name, file_name), 'wb') as f:
                    pickle.dump(champ_net, f)
        except KeyboardInterrupt:
            print("User break.")
            break
    env.close()


def continue_run(filename):
    """Continues run, based on filename of neat-checkpoint."""
    # Load the configuration file, which is assumed to live in the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    # Restore population.
    population = neat.checkpoint.Checkpointer.restore_checkpoint(filename)
    # Override configuration.
    population.config = config
    # Run population.
    run(population)


def start_run():
    """Starts run with new population."""
    # Load the configuration file, which is assumed to live in the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    population = neat.Population(config)
    # Starts run with population.
    run(population)


if __name__ == '__main__':
    start_run()
