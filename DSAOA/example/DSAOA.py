import numpy as np
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.backends import cudnn
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from layout_data.models.model import UnetSL
from layout_data.utils.options import parses_ul


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
    # 确保输入是整数数组
    list_arr = np.array(solution_list, dtype=int).flatten()
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

    # 这一步排序在生成Mask时其实不是必须的，但保留你的逻辑
    sorted_indices = np.lexsort((position[:, 1], position[:, 0]))
    return position[sorted_indices]


def forcing(xs, ys, solution_list):
    """
    将方案编号列表转换为物理场矩阵 (Input Channel)
    """
    position = locate_heat_source(solution_list)
    N = len(xs)
    M = len(ys)

    # 如果没有组件，返回全零
    if position.size == 0:
        return np.zeros((N, M), dtype=np.float32)

    len_heatsource = 0.01
    Phi = 10000
    heat_k = 1
    value_to_set = -Phi / heat_k

    fmat = np.zeros((N, M), dtype=np.float32)

    # 使用 numpy 的广播机制优化 meshgrid，减少内存消耗
    # 注意：这里为了保持和你原代码一致，使用了 indexing='ij'
    xx, yy = np.meshgrid(xs, ys, indexing='ij')

    for k in range(position.shape[0]):
        x_start = position[k, 0]
        y_start = position[k, 1]
        x_end = x_start + len_heatsource
        y_end = y_start + len_heatsource

        # 利用 Boolean Indexing
        mask = (xx >= x_start) & (yy >= y_start) & (xx <= x_end) & (yy <= y_end)
        fmat[mask] = value_to_set

    return fmat


def evaluate_batch_solutions(solutions_list, xs, ys, model, device='cuda'):
    """
        评估
    """
    batch_layouts = []
    # 如果输入是单个 numpy 数组 (Batch, k)，转为 list 迭代
    if isinstance(solutions_list, np.ndarray) and solutions_list.ndim == 2:
        iterable = solutions_list
    else:
        iterable = solutions_list

    for sol in iterable:
        layout = forcing(xs, ys, sol)
        batch_layouts.append(layout)

    # 转换为 Tensor Batch: [Batch, 1, H, W]
    batch_tensor = torch.tensor(np.array(batch_layouts), dtype=torch.float32)
    if batch_tensor.ndim == 3:
        batch_tensor = batch_tensor.unsqueeze(1)  # 增加 Channel 维度

    batch_tensor = batch_tensor.to(device)

    # 模型批量推理
    model.eval()
    with torch.no_grad():
        heat_pre = model(batch_tensor)
        heat_pre = heat_pre + 298

        max_temps, _ = torch.max(heat_pre.view(heat_pre.size(0), -1), dim=1)

    return max_temps.cpu().numpy()


def DSAOA(k, popnum, max_iterations, hparams):
    print(f'算法开始，元件数量为 {k}，种群大小 {popnum}')

    # --- 初始化环境 ---
    seed = hparams.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    print("正在加载模型...")
    model = UnetSL(hparams).to(device)

    model_path = '/example/lightning_logs/version_0/checkpoints/epoch=29-step=20249.ckpt'

    if not os.path.exists(model_path):
        print(f"警告: 模型路径 {model_path} 不存在，请检查路径。")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
    except Exception as e:
        print(f"模型加载失败: {e}")
        return []

    model.eval()

    # 坐标定义
    J, K, N, M = 0.1, 0.1, 200, 200
    xx = np.linspace(0, J, N)
    yy = np.linspace(0, K, M)
    ys = y_refine_function(yy)
    xs = x_refine_function(xx)
    all_numbers = np.arange(1, 101)

    # 初始随机生成种群

    pop = np.zeros((popnum, k), dtype=int)
    for i in range(popnum):
        pop[i, :] = np.sort(np.random.choice(all_numbers, k, replace=False))

    # 批量评估初始种群
    pop_fitness = evaluate_batch_solutions(pop, xs, ys, model, device)

    # 记录历史
    min_idx = np.argmin(pop_fitness)
    global_best_score = pop_fitness[min_idx]
    global_best_sol = pop[min_idx, :].copy()
    score_history = [global_best_score]

    print(f'初始最佳得分: {global_best_score:.4f}')

    # --- 2. 迭代 ---
    for iter_count in range(1, max_iterations + 1):
        start_time = time.time()
        has_improved_in_generation = False
        for num in range(popnum):
            current_solution = pop[num, :]
            current_fitness = pop_fitness[num]

            # 一次生成 k 个残缺方案
            incomplete_candidates = np.zeros((k, k - 1), dtype=int)
            for i in range(k):
                incomplete_candidates[i, :] = np.delete(current_solution, i)

            # 批量评估残缺方案 (Batch = k)
            incomplete_scores = evaluate_batch_solutions(incomplete_candidates, xs, ys, model, device)

            # 找最好的残缺
            best_inc_idx = np.argmin(incomplete_scores)
            best_inc_sol = incomplete_candidates[best_inc_idx, :]

            candidate_pool = np.setdiff1d(all_numbers, best_inc_sol)
            num_candidates = len(candidate_pool)

            new_candidates = np.tile(best_inc_sol, (num_candidates, 1))  # (81, k-1)

            new_candidates = np.hstack((new_candidates, candidate_pool.reshape(-1, 1)))  # (81, k)
            # 保持方案的一致性
            new_candidates.sort(axis=1)

            # 批量评估所有新方案
            new_scores = evaluate_batch_solutions(new_candidates, xs, ys, model, device)

            best_new_idx = np.argmin(new_scores)
            best_new_score = new_scores[best_new_idx]
            best_new_sol = new_candidates[best_new_idx, :]

            # 更新个体
            if best_new_score < current_fitness:
                pop[num, :] = best_new_sol
                pop_fitness[num] = best_new_score
                has_improved_in_generation = True

                # 实时更新全局最优
                if best_new_score < global_best_score:
                    global_best_score = best_new_score
                    global_best_sol = best_new_sol.copy()

        # 每一代结束记录一次
        score_history.append(global_best_score)
        elapsed = time.time() - start_time
        print(f'Iter {iter_count}/{max_iterations} | Best Score: {global_best_score:.4f} | Time: {elapsed:.2f}s')

        if not has_improved_in_generation:
            print("本代无个体提升，算法收敛。")
            break

    return score_history, global_best_sol


if __name__ == "__main__":
    config_path = Path(__file__).absolute().parent / "config_l.yml"
    hparams = parses_ul(config_path)
    history, best_layout = DSAOA(k=20, popnum=5, max_iterations=10, hparams=hparams)
    print("优化历史:", history)
    print(best_layout)
