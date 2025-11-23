import random
import numpy as np
import datetime
from osmoea.run import run_nsga_ii

if __name__ == "__main__":
    # seed = 0
    # random.seed(seed)
    # np.random.seed(seed)

    start = datetime.datetime.now()
    run_nsga_ii(
        experiment_name_1="osmoea_1",
        structure_shape = (5, 5),
        pop_size=25,
        max_evaluations =250,
        train_iters_1 = 500,
        train_iters_2 = 750,
        num_cores = 10,
        env_name_1 = 'Walker-v0',
        env_name_2 = 'BridgeWalker-v0',
    )
    end = datetime.datetime.now()
    print(start)
    print(end)
