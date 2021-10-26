import gym
import numpy as np
import time
from gym.wrappers import FrameStack, AtariPreprocessing
import gc
import pickle
import os
import argparse
import tensorflow as tf

def runExperiment(domain, save_metric_dir, trialNum, model_save_dir, learningMode, novelty_def):
    import sys
    sys.path.append('..')
    from src.policies.widthLookahead import IWRollout
    for checkdir in [save_metric_dir, model_save_dir]:
        filename = '{}/'.format(checkdir)
        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

    env = gym.make('{}-v4'.format(domain)).env
    env.unwrapped.frameskip = 15

    env = FrameStack(AtariPreprocessing(env, noop_max=0, frame_skip=1, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True,
                     grayscale_newaxis=False, scale_obs=False),4)
    rewards = []
    metrics = []

    if learningMode == "crit":
        useCritPath = True
        useAll = False
        useValueFunction = True
    elif learningMode == "all":
        useCritPath = False
        useAll = True
        useValueFunction = True
    elif learningMode == "None":
        useCritPath = False
        useAll = False
        useValueFunction = False
    else:
        useCritPath = True
        useAll = False
        useValueFunction = False

    params = {  'sim_budget': 100,
                'rolloutHorizon': 100,
                'noveltyType': novelty_def,
                 'cache': True,
                 'num_actions': env.action_space.n,
                 'gamma': 1.00,
                 'model_save_dir': model_save_dir,
                'useCritPath': useCritPath,
                'useAll': useAll,
                'useValueFunction': useValueFunction
                 }


    iwRollout = IWRollout(**params)
    while (learningMode != "None"  or  len(rewards) < 10) and iwRollout.sim_calls < 22000000:
        startTime = time.time()
        trained = iwRollout.resetEp(rewards) # Checks if it is time to check if to update networks.
        timeReset = time.time() - startTime
        timeEval = 0
        gc.collect()
        state = np.array(env.reset())
        done = False
        T_reward = 0
        step = 0
        horizon = 1200
        while not done and step < horizon:
            step += 1
            startTime = time.time()
            action, Qs, reward, done, state = iwRollout.get_action(env, state)
            timeEval += time.time() - startTime
            T_reward += reward

        if T_reward is not None:
            rewards.append(T_reward)
            metrics.append((T_reward, step, timeReset, timeEval, trained, iwRollout.sim_calls))

        with open('{}'.format(os.path.join('{}'.format(
                save_metric_dir),
                '{}-trial-{}.evaluate'.format(
                    domain,
                    trialNum))), 'wb') as output:
            pickle.dump(metrics, output, pickle.HIGHEST_PROTOCOL)

    del iwRollout
    tf.keras.backend.clear_session()
    gc.collect()


if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--domain', default="Breakout")
        parser.add_argument('--novelty_def', default='depth')
        parser.add_argument('--learningMode', default='crit')
        parser.add_argument('--trialNum', default=0)

        args = parser.parse_args()
        runExperiment(args.domain, "../results/{}_crit{}_allGames/{}/".format(args.novelty_def, args.learningMode, args.domain),
                             args.trialNum,
                             "../results/savedNetworks_{}_crit{}_allGames/{}/models/{}".format(args.novelty_def,
                                                                                                          args.learningMode,
                                                                                                          args.domain,
                                                                                                          args.trialNum),
                             args.learningMode, args.novelty_def)

        with open('../forParallelRuns/{}-{}-{}-trial-{}.finished'.format(
                    args.novelty_def,
                    args.learningMode,
                    args.domain,
                    args.trialNum), 'wb') as output:
            pickle.dump("finished", output, pickle.HIGHEST_PROTOCOL)
