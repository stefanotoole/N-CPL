import time
import os
import sys

def main(cmd_line_args) :
    DOMAINS = ['alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis',
    'bank_heist', 'battle_zone', 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout',
    'centipede', 'chopper_command', 'crazy_climber', 'demon_attack', 'double_dunk', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar',
    'hero', 'ice_hockey', 'jamesbond', 'kangaroo', 'krull', 'kung_fu_master',
    'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong',
    'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing',
    'solaris', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down',
    'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon']
    trialNums = [0, 1, 2, 3, 4]
    num_workers = 80
    DIR = '../forParallelRuns'
    workersLaunched = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    useCrit = ["None"]
    novelty = ["depth", "classic"]
    for trialNum in trialNums:
        for game in DOMAINS:
            domain = ''.join([g.capitalize() for g in game.split('_')])
            for learningMode in useCrit:
                for novelty_def in novelty:
                    launched = False
                    while not launched:
                        finishedJobs = len(
                                [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
                        print("No. finished jobs is {} number launched {}".format(finishedJobs, num_workers))
                        if os.path.isfile('../forParallelRuns/{}-{}-{}-trial-{}.finished'.format(novelty_def, learningMode, domain, trialNum)):
                            print("already exists")
                            launched = True
                        elif workersLaunched < num_workers + finishedJobs:
                            os.system(sys.executable + " RIW_atari_single.py --domain {} --novelty_def {} --learningMode {} --trialNum {} &".format(
                                domain, novelty_def, learningMode, trialNum))
                            time.sleep(5)
                            workersLaunched += 1
                            launched = True
                        else:
                            time.sleep(120) # Every 2 mins check if previous workers have finished
    print("launched all workers")

if __name__ == "__main__":
        main(sys.argv[1:])
