import os
import numpy as np
import shutil
import random
import math
import time
import sys
import torch
import torch.nn.functional as F
from torch.backends import cudnn
from pathlib import Path
from openai import OpenAI
# 路径设置
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
root_dir = os.path.join(curPath, '..')
if rootPath not in sys.path:
    sys.path.append(rootPath)

from layout_data.models.model import UnetSL
from layout_data.utils.options import parses_ul


def parser(content):
    try:
        # 这里假设 LLM 返回的是 [[0,1...], ... ] 格式
        start = content.find('[')
        end = content.rfind(']') + 1
        if start == -1 or end == 0: return np.zeros((10, 10))
        arr_str = content[start:end]
        return np.array(eval(arr_str))
    except:
        return np.zeros((10, 10))


# --- OpenAI 设置 ---
os.environ['OPENAI_API_KEY'] = "YOUR_KEY_HERE"  # 请填写 KEY
os.environ['OPENAI_BASE_URL'] = "YOUR_URL_HERE"  # 请填写 URL
client = OpenAI()

# --------------------------------------------------------------------------
# 1. 修正后的 Prompt (针对热源布局优化，而非机器人)
# --------------------------------------------------------------------------
system_prompt = """You are an intelligent search operator in an Evolutionary Algorithm optimizing a 2D Heat Source Layout.
Your goal is to propose a new layout design that minimizes/optimizes the thermal performance (fitness).
The layout is a 10x10 grid. Each cell is either 0 (empty) or 1 (heat source).
You will be given existing high-performing solutions sorted by fitness.
Output a new solution starting with <begin> and ending with <end>. The format should be a python list of lists representing the 10x10 matrix.
Only generate the new solution. No explanation."""

tlo_description = """We are designing a 2D Heat Source Layout on a 10x10 grid.
The goal is to place a fixed number of heat components (represented as '1') on the grid to manage the temperature field effectively.
Empty spaces are represented as '0'.
Positions are indexed from 1 to 100, or row/col (0-9).
"""

task_description = """The objective is to optimize the layout to achieve a specific thermal metric (Fitness).
Based on the provided examples, try to understand the pattern of component placement that leads to better fitness scores.
"""

constraint_description = """1. The layout must be a 10x10 matrix.
2. The number of heat components (1s) is fixed (e.g., 20 components). Ensure the new solution has exactly the same number of 1s as the examples.
"""


class Structure:
    def __init__(self, list_data, layout, label, fitness):
        self.list = list_data  # 组件编号列表 [1, 15, ...]
        self.layout = layout  # 10x10 矩阵
        self.label = label  # 标签 string
        self.fitness = fitness  # float 分数
        self.field = None  # 存储热场 (可选)
        self.is_survivor = False
        self.prev_gen_label = None


def x_refine_function(in_vec):
    A = in_vec[0]
    B = in_vec[-1]
    term1 = 4 / ((B - A) ** 2)
    term2 = (in_vec - (B + A) / 2) ** 3
    term3 = (A + B) / 2
    out = term1 * term2 + term3
    return out


def y_refine_function(in_vec):
    A = in_vec[0]
    B = in_vec[-1]
    numerator = (in_vec ** 2) + (A * B)
    denominator = A + B
    out = numerator / denominator
    return out


def locate_heat_source(solution_list):
    list_arr = np.array(solution_list).flatten()
    n = len(list_arr)
    position = np.zeros((n, 2))
    for i in range(n):
        list_i = list_arr[i]
        row = list_i // 10
        col = list_i - row * 10
        if col == 0:
            row = row - 1
            col = 10
        x = (col - 1) * 0.01
        y = row * 0.01
        position[i] = [x, y]
    sorted_indices = np.lexsort((position[:, 1], position[:, 0]))
    return position[sorted_indices]


def forcing(xs, ys, solution_list):
    position = locate_heat_source(solution_list)
    if position.size == 0:
        return np.zeros((len(xs), len(ys)))

    len_heatsource = 0.01
    Phi = 10000
    heat_k = 1
    value_to_set = -Phi / heat_k

    N = len(xs)
    M = len(ys)
    fmat = np.zeros((N, M))
    xx, yy = np.meshgrid(xs, ys, indexing='ij')

    for k in range(position.shape[0]):
        x_start = position[k, 0]
        y_start = position[k, 1]
        x_end = x_start + len_heatsource
        y_end = y_start + len_heatsource
        mask = (xx >= x_start) & (yy >= y_start) & (xx <= x_end) & (yy <= y_end)
        fmat[mask] = value_to_set
    return fmat


def evaluate_batch_solutions(solutions_list, xs, ys, model, device='cuda'):
    """
    批量将 list 形式的解转换为热图并推断
    """
    if len(solutions_list) == 0:
        return []
    batch_layouts = []
    # 判断输入是 list of lists 还是 numpy array
    for sol in solutions_list:
        layout = forcing(xs, ys, sol)
        batch_layouts.append(layout)

    # 转换为 Tensor Batch: [Batch, 1, H, W]
    batch_tensor = torch.tensor(np.array(batch_layouts), dtype=torch.float32)
    if batch_tensor.ndim == 3:
        batch_tensor = batch_tensor.unsqueeze(1)

    batch_tensor = batch_tensor.to(device)

    model.eval()
    with torch.no_grad():
        heat_pre = model(batch_tensor)
        heat_pre = heat_pre + 298
        max_temps, _ = torch.max(heat_pre.view(heat_pre.size(0), -1), dim=1)

    return max_temps.cpu().numpy()


def list2layout(num_list, layout_shape):
    matrix = np.zeros(layout_shape, dtype=int)
    for number in num_list:
        if 1 <= number <= 100:
            row_idx = (number - 1) // 10
            col_idx = (number - 1) % 10
            if row_idx < layout_shape[0] and col_idx < layout_shape[1]:
                matrix[row_idx][col_idx] = 1
    return matrix


def matrix2list(matrix):
    num_list = []
    rows, cols = matrix.shape
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == 1:
                number = r * 10 + c + 1
                num_list.append(number)
    num_list.sort()
    return num_list


def mutate(parent, num_attempts):
    # 简单的单点变异
    if num_attempts >= len(parent):
        return parent

    new_list = parent.copy()
    all_pos = set(range(1, 101))
    current_pos = set(parent)
    available = list(all_pos - current_pos)

    if len(available) < num_attempts:
        return parent

    to_remove = random.sample(range(len(new_list)), num_attempts)
    to_add = random.sample(available, num_attempts)

    for i, idx in enumerate(to_remove):
        new_list[idx] = to_add[i]

    return sorted(new_list)


def run(experiment_name, layout_shape, components, pop_size, max_evaluations, hparams):

    home_path = os.path.join(root_dir, "saved_data", experiment_name)
    if os.path.exists(home_path):
        print(f'Experiment {experiment_name} exists. Overwrite? (y/n): ', end="")
        if input().lower() == 'y':
            shutil.rmtree(home_path)
        else:
            return
    os.makedirs(home_path, exist_ok=True)

    print("Loading Model...")
    seed = hparams.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UnetSL(hparams).to(device)

    model_path = '/example/lightning_logs/version_0/checkpoints/epoch=80.ckpt'
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print(f"Error: Checkpoint not found at {model_path}")
        return
    model.eval()

    # --- 坐标定义 ---
    xx = np.linspace(0, 0.1, 200)
    yy = np.linspace(0, 0.1, 200)
    ys = y_refine_function(yy)
    xs = x_refine_function(xx)

    # --- 初始化种群 ---
    print("Initializing Population...")
    layouts = []  # List of Structure objects

    # 批量生成初始解
    initial_pop_lists = []
    for i in range(pop_size):
        temp_list = sorted(random.sample(range(1, 101), components))
        initial_pop_lists.append(temp_list)

    initial_fitness = evaluate_batch_solutions(initial_pop_lists, xs, ys, model, device)

    for i in range(pop_size):
        temp_list = initial_pop_lists[i]
        temp_layout = list2layout(temp_list, layout_shape)
        layouts.append(Structure(temp_list, temp_layout, f"Gen0-{i}", initial_fitness[i]))

    population_hash_set = set()
    for l in layouts:
        population_hash_set.add(tuple(l.list))

    generation = 0
    num_evals_total = pop_size

    while generation < 100 and num_evals_total < max_evaluations:
        print(f"\n--- Generation {generation} ---")
        # 1. 排序
        layouts.sort(key=lambda x: x.fitness)  # 假设 fitness 越小越好? (温度越低越好?)
        # 2. 存活者选择 (Elitism)
        num_survivors = max(2, int(pop_size * 0.5))
        survivors = layouts[:num_survivors]  # 保留前 50%

        # 重置当前代种群，加入幸存者
        next_gen_layouts = []
        for s in survivors:
            s.is_survivor = True
            next_gen_layouts.append(s)

        # 3. 准备生成新个体 (GA 或 LLM)
        new_candidates_list = []  # 暂存新生成的 list，稍后批量评估
        new_candidates_sources = []  # 记录来源 'GA' or 'LLM'

        # 早期阶段: 纯 GA (Generation < 4)
        if generation < 4:
            needed = pop_size - len(next_gen_layouts)
            attempts = 0
            while len(new_candidates_list) < needed and attempts < needed * 5:
                attempts += 1
                parent = random.choice(survivors)
                child_list = mutate(parent.list, num_attempts=2)

                if tuple(child_list) not in population_hash_set:
                    new_candidates_list.append(child_list)
                    new_candidates_sources.append("GA")
                    population_hash_set.add(tuple(child_list))

        # LLM 阶段 (Generation >= 4)
        else:
            shots_indices = np.linspace(0, len(survivors) - 1, 5, dtype=int)
            examples_str = ""
            for idx in shots_indices:
                s = survivors[idx]
                examples_str += f"{s.list}, {s.fitness:.3f}; "

            best_val = survivors[0].fitness
            target_val = best_val * 0.95

            prompt_content = tlo_description + constraint_description + \
                             f"Examples (Solution, Fitness): {examples_str}\n" + \
                             f"Target Fitness: {target_val:.3f}\n" + \
                             "Generate a new valid 10x10 matrix layout."

            llm_attempts = 0
            llm_success = 0
            needed = pop_size - len(next_gen_layouts)

            while len(new_candidates_list) < needed and llm_attempts < 10:
                print(f"Querying LLM (Attempt {llm_attempts})...")
                try:
                    response = client.chat.completions.create(
                        model="deepseek-R1-671B",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt_content}
                        ],
                        temperature=1.2,
                        max_tokens=1024
                    )
                    content = response.choices[0].message.content
                    matrix = parser(content)
                    child_list = matrix2list(matrix)

                    if len(child_list) == components and tuple(child_list) not in population_hash_set:
                        is_too_similar = False
                        for exist_sol in [s.list for s in survivors]:
                            intersection = len(set(child_list) & set(exist_sol))
                            if intersection > components - 2:
                                is_too_similar = True
                                break

                        if not is_too_similar:
                            new_candidates_list.append(child_list)
                            new_candidates_sources.append("LLM")
                            population_hash_set.add(tuple(child_list))
                            llm_success += 1
                            print("LLM generated valid solution.")
                except Exception as e:
                    print(f"LLM Error: {e}")

                llm_attempts += 1

            while len(new_candidates_list) < needed:
                parent = random.choice(survivors)
                child_list = mutate(parent.list, num_attempts=random.randint(1, 4))
                if tuple(child_list) not in population_hash_set:
                    new_candidates_list.append(child_list)
                    new_candidates_sources.append("GA-Fill")
                    population_hash_set.add(tuple(child_list))

        if len(new_candidates_list) > 0:
            print(f"Evaluating {len(new_candidates_list)} new candidates...")
            scores = evaluate_batch_solutions(new_candidates_list, xs, ys, model, device)

            for idx, sc in enumerate(scores):
                cand_list = new_candidates_list[idx]
                cand_src = new_candidates_sources[idx]
                cand_layout = list2layout(cand_list, layout_shape)

                new_struct = Structure(cand_list, cand_layout, f"Gen{generation}-{cand_src}-{idx}", sc)
                next_gen_layouts.append(new_struct)
                num_evals_total += 1

        layouts = next_gen_layouts

        save_path = os.path.join(home_path, f"gen_{generation}")
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "scores.txt"), "w") as f:
            for l in layouts:
                f.write(f"{l.label}: {l.fitness}\n")

        print(f"Gen {generation} Best Fitness: {layouts[0].fitness:.4f}")
        generation += 1


if __name__ == "__main__":
    config_path = Path(__file__).absolute().parent / "config_l.yml"
    hparams = parses_ul(config_path)
    run(
        experiment_name="HeatOpt_Experiment_1",
        layout_shape=(10, 10),
        components=20,
        pop_size=50,
        max_evaluations=1000,
        hparams=hparams
    )