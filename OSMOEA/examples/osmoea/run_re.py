import os
import numpy as np
import shutil
import random
import math

import sys
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '..')
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))

from ppo import run_ppo
from evogym import hashable, get_full_connectivity
import utils.mp_group as mp
from utils.algo_utils import TerminationCondition, Structure

def run_reward(experiment_name, robot_structure, train_iters, num_cores, env_name):

    ### STARTUP: ANAGE DIRECTORIES ###
    home_path = os.path.join(root_dir, "saved_data", experiment_name)
    start_gen = 0

    ### DEFINE TERMINATION CONDITION ###    
    tc = TerminationCondition(train_iters)

    is_continuing = False    
    try:
        os.makedirs(home_path)
    except:
        print(f'THIS EXPERIMENT ({experiment_name}) ALREADY EXISTS')
        print("Override? (y/n/c): ", end="")
        ans = input()
        if ans.lower() == "y":
            shutil.rmtree(home_path)
            print()
        elif ans.lower() == "c":
            print("Enter gen to start training on (0-indexed): ", end="")
            start_gen = int(input())
            is_continuing = True
            print()
        else:
            return

    ### STORE META-DATA ##
    if not is_continuing:
        temp_path = os.path.join(root_dir, "saved_data", experiment_name, "metadata.txt")
        
        try:
            os.makedirs(os.path.join(root_dir, "saved_data", experiment_name))
        except:
            pass

        f = open(temp_path, "w")
        f.write(f'ROBOT_STRUCTURE: {robot_structure}\n')
        f.write(f'ENV_NAME: {env_name}\n')
        f.write(f'TRAIN_ITERS: {train_iters}\n')
        f.close()

    else:
        temp_path = os.path.join(root_dir, "saved_data", experiment_name, "metadata.txt")
        f = open(temp_path, "r")
        count = 0
        for line in f:
            if count == 0:
                pop_size = int(line.split()[1])
            if count == 1:
                structure_shape = (int(line.split()[1]), int(line.split()[2]))
            if count == 2:
                max_evaluations = int(line.split()[1])
            if count == 3:
                train_iters = int(line.split()[1])
                tc.change_target(train_iters)
            count += 1

        print(f'Starting training with pop_size {pop_size}, shape ({structure_shape[0]}, {structure_shape[1]}), ' + 
            f'max evals: {max_evaluations}, train iters {train_iters}.')
        
        f.close()

    ### GENERATE // GET INITIAL POPULATION ###
    structures = []
    population_structure_hashes = {}
    generation = 0
    
    #generate a population
    if not is_continuing: 
        robot = np.array(robot_structure)
        temp_structure = (robot, get_full_connectivity(robot))
        structures.append(Structure(*temp_structure, i))
        population_structure_hashes[hashable(temp_structure[0])] = True
    while True:
        ### MAKE GENERATION DIRECTORIES ###
        save_path_structure = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "structure")
        save_path_controller = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "controller")
        # save_path_history = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "history")

        try:
            os.makedirs(save_path_structure)
        except:
            pass

        try:
            os.makedirs(save_path_controller)
        except:
            pass

        # try:
        #     os.makedirs(save_path_history)
        # except:
        #     pass


        ### SAVE POPULATION DATA ###
        for i in range(len(structures)):
            temp_path = os.path.join(save_path_structure, str(structures[i].label))
            np.savez(temp_path, structures[i].body, structures[i].connections)

        ### TRAIN GENERATION
        #better parallel
        group = mp.Group()
        for structure in structures:

            if structure.is_survivor:
                save_path_controller_part = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "controller",
                    "robot_" + str(structure.label) + "_controller" + ".pt")
                save_path_controller_part_old = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation-1), "controller",
                    "robot_" + str(structure.prev_gen_label) + "_controller" + ".pt")
                print(f'Skipping training for {save_path_controller_part}.\n')
                try:
                    shutil.copy(save_path_controller_part_old, save_path_controller_part)
                except:
                    print(f'Error coppying controller for {save_path_controller_part}.\n')

                # try:
                #     shutil.copy(save_path_history_part_old, save_path_history_part)
                # except:
                #     print(f'Error coppying history rewards for {save_path_history_part}.\n')
            else:
                ppo_args = ((structure.body, structure.connections), tc, (save_path_controller, structure.label), env_name)
                group.add_job(run_ppo, ppo_args, callback=structure.set_reward)

        group.run_jobs(num_cores)

        ### SAVE histroy rewards
        # for i in range(len(structures)):
        #     temp_history_path = os.path.join(save_path_history, "robot_" + str(structures[i].label)+ ".npz")
        #     history_rewards = np.load(temp_history_path)['arr_0']
        #     structures[i].history = history_rewards

        #not parallel
        #for structure in structures:
        #    ppo.run_algo(structure=(structure.body, structure.connections), termination_condition=termination_condition, saving_convention=(save_path_controller, structure.label))

        ### COMPUTE FITNESS, SORT, AND SAVE ###
        for structure in structures:
            structure.compute_fitness()

        structures = sorted(structures, key=lambda structure: structure.fitness, reverse=True)

        #SAVE RANKING TO FILE
        temp_path = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "output.txt")
        f = open(temp_path, "w")

        out = ""
        for structure in structures:
            out += str(structure.label) + "\t\t" + str(structure.fitness) + "\n"
        f.write(out)
        f.close()
        return


