import os
import numpy as np
import shutil
import random
import math
import functools

import sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '..')
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))

from ppo import run_ppo
from evogym import sample_robot, hashable
import utils.mp_group as mp
from utils.algo_utils import get_percent_survival_evals, mutate, TerminationCondition, Structure


# ==========================================
# NSGA-II 核心辅助函数 (保持不变)
# ==========================================

def dominates(p_fitness, q_fitness):
    """判断 p 是否支配 q (最大化问题)"""
    better_in_one = False
    for f1, f2 in zip(p_fitness, q_fitness):
        if f1 < f2: return False
        if f1 > f2: better_in_one = True
    return better_in_one


def fast_non_dominated_sort(structures):
    """快速非支配排序"""
    fronts = [[]]
    for p in structures:
        p.domination_count = 0
        p.dominated_solutions = []
        for q in structures:
            if p == q: continue
            if dominates(p.fitnesses, q.fitnesses):
                p.dominated_solutions.append(q)
            elif dominates(q.fitnesses, p.fitnesses):
                p.domination_count += 1
        if p.domination_count == 0:
            p.rank = 0
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in p.dominated_solutions:
                q.domination_count -= 1
                if q.domination_count == 0:
                    q.rank = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    if len(fronts[-1]) == 0: fronts.pop()
    return fronts


def calculate_crowding_distance(front):
    """计算拥挤距离"""
    l = len(front)
    if l == 0: return
    for p in front: p.crowding_distance = 0
    num_objectives = len(front[0].fitnesses)
    for m in range(num_objectives):
        front.sort(key=lambda x: x.fitnesses[m])
        front[0].crowding_distance = float('inf')
        front[-1].crowding_distance = float('inf')
        f_min, f_max = front[0].fitnesses[m], front[-1].fitnesses[m]
        if f_max == f_min: continue
        for i in range(1, l - 1):
            front[i].crowding_distance += (front[i + 1].fitnesses[m] - front[i - 1].fitnesses[m]) / (f_max - f_min)


def nsga_ii_compare(s1, s2):
    """NSGA-II 比较算子"""
    if s1.rank < s2.rank:
        return -1
    elif s1.rank > s2.rank:
        return 1
    else:
        if s1.crowding_distance > s2.crowding_distance:
            return -1
        elif s1.crowding_distance < s2.crowding_distance:
            return 1
        else:
            return 0


def run_nsga_ii(experiment_name_1, structure_shape, pop_size, max_evaluations, train_iters_1, train_iters_2, num_cores,
                env_name_1, env_name_2):

    experiment_name = experiment_name_1
    env_names = [env_name_1, env_name_2]

    train_iters_list = [train_iters_1, train_iters_2]

    print(f"\nStarting NSGA-II.")
    print(f"Task 1: {env_name_1} (Target Iters: {train_iters_1})")
    print(f"Task 2: {env_name_2} (Target Iters: {train_iters_2})")

    ### STARTUP: MANAGE DIRECTORIES ###
    home_path = os.path.join(root_dir, "saved_data", experiment_name)
    start_gen = 0

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

    if not is_continuing:
        temp_path = os.path.join(root_dir, "saved_data", experiment_name, "metadata.txt")
        try:
            os.makedirs(os.path.join(root_dir, "saved_data", experiment_name))
        except:
            pass
        f = open(temp_path, "w")
        f.write(f'POP_SIZE: {pop_size}\n')
        f.write(f'STRUCTURE_SHAPE: {structure_shape[0]} {structure_shape[1]}\n')
        f.write(f'MAX_EVALUATIONS: {max_evaluations}\n')
        # 记录两个 Iterations
        f.write(f'TRAIN_ITERS: {train_iters_1} {train_iters_2}\n')
        f.write(f'TASKS: {env_name_1},{env_name_2}\n')
        f.close()
    else:

        pass

    structures = []
    population_structure_hashes = {}
    num_evaluations = 0
    generation = 0

    if not is_continuing:
        for i in range(pop_size):
            temp_structure = sample_robot(structure_shape)
            while (hashable(temp_structure[0]) in population_structure_hashes):
                temp_structure = sample_robot(structure_shape)

            structures.append(Structure(*temp_structure, i))
            structures[-1].fitnesses = [None] * len(env_names)
            population_structure_hashes[hashable(temp_structure[0])] = True
            num_evaluations += 1

    while True:
        percent_survival = get_percent_survival_evals(num_evaluations, max_evaluations)
        num_survivors = max(2, math.ceil(pop_size * percent_survival))

        gen_dir = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation))
        save_path_structure = os.path.join(gen_dir, "structure")
        save_path_controller = os.path.join(gen_dir, "controller")

        try:
            os.makedirs(save_path_structure)
            os.makedirs(save_path_controller)
        except:
            pass

        for i in range(len(structures)):
            temp_path = os.path.join(save_path_structure, str(structures[i].label))
            np.savez(temp_path, structures[i].body, structures[i].connections)

        current_iters_list = [int(iters * fidelity_scale) for iters in train_iters_list]

        print(f"Generation {generation}: Scale {fidelity_scale:.2f}. Training iters: {current_iters_list}")

        group = mp.Group()

        def make_reward_callback(struct_obj, task_idx):
            def callback(reward):
                if not hasattr(struct_obj, 'fitnesses') or struct_obj.fitnesses is None:
                    struct_obj.fitnesses = [None] * len(env_names)
                struct_obj.fitnesses[task_idx] = reward

            return callback

        for structure in structures:
            if structure.is_survivor and hasattr(structure, 'fitnesses') and None not in structure.fitnesses:
                for task_idx in range(len(env_names)):
                    old_ctrl = os.path.join(root_dir, "saved_data", experiment_name,
                                            "generation_" + str(generation - 1), "controller",
                                            f"robot_{structure.prev_gen_label}_task_{task_idx}_controller.pt")
                    new_ctrl = os.path.join(save_path_controller,
                                            f"robot_{structure.label}_task_{task_idx}_controller.pt")
                    if os.path.exists(old_ctrl):
                        shutil.copy(old_ctrl, new_ctrl)
            else:
                structure.fitnesses = [None] * len(env_names)
                for task_idx, env_name in enumerate(env_names):
                    task_iters = current_iters_list[task_idx]
                    current_tc = TerminationCondition(task_iters)
                    ppo_save_prefix = f"robot_{structure.label}_task_{task_idx}"  # run_ppo 内部会添加 _controller.pt
                    ppo_args = (
                        (structure.body, structure.connections),
                        current_tc,
                        (save_path_controller, ppo_save_prefix),
                        env_name
                    )

                    group.add_job(run_ppo, ppo_args, callback=make_reward_callback(structure, task_idx))

        group.run_jobs(num_cores)

        valid_structures = []
        for s in structures:
            if s.fitnesses is None: s.fitnesses = [-10000.0] * len(env_names)
            s.fitnesses = [f if f is not None else -10000.0 for f in s.fitnesses]
            s.fitness = np.mean(s.fitnesses)
            valid_structures.append(s)
        structures = valid_structures
        fronts = fast_non_dominated_sort(structures)
        for front in fronts:
            calculate_crowding_distance(front)
        structures = sorted(structures, key=functools.cmp_to_key(nsga_ii_compare))
        temp_path = os.path.join(gen_dir, "output.txt")
        with open(temp_path, "w") as f:
            out = f"Label\tRank\tDist\tFitnesses ({env_names})\n"
            for s in structures:
                fit_str = "\t".join([f"{x:.2f}" for x in s.fitnesses])
                out += f"{s.label}\t{s.rank}\t{s.crowding_distance:.2f}\t{fit_str}\n"
            f.write(out)
        if num_evaluations >= max_evaluations:
            print(f'Trained {num_evaluations} robots. Done.')
            return

        print(f'FINISHED GENERATION {generation} - Top of Pareto Front:\n')
        if len(fronts) > 0:
            for p in fronts[0][:3]:
                print(f"ID: {p.label}, Fits: {['{:.2f}'.format(x) for x in p.fitnesses]}")
        survivors = structures[:num_survivors]
        for s in structures: s.is_survivor = False
        for i, s in enumerate(survivors):
            s.is_survivor = True
            s.prev_gen_label = s.label
            s.label = i

        next_gen_structures = survivors[:]
        num_children = 0
        target_children = pop_size - num_survivors
        while num_children < target_children and num_evaluations < max_evaluations:
            candidates = random.sample(survivors, 2)
            parent = candidates[0] if nsga_ii_compare(candidates[0], candidates[1]) < 0 else candidates[1]
            child_data = mutate(parent.body.copy(), mutation_rate=0.1, num_attempts=50)
            if child_data != None and hashable(child_data[0]) not in population_structure_hashes:
                new_struct = Structure(*child_data, num_survivors + num_children)
                new_struct.fitnesses = [None] * len(env_names)
                next_gen_structures.append(new_struct)
                population_structure_hashes[hashable(child_data[0])] = True
                num_children += 1
                num_evaluations += 1
        structures = next_gen_structures
        generation += 1

