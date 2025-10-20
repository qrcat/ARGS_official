from utils.io import activated_gs2train_gs
from pgs import PGSMoments
import numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing
import argparse
import pickle
import time
import copy

parser = argparse.ArgumentParser()
# inputs
parser.add_argument("--input", default="data/airplane_enhance")
# outputs
parser.add_argument("--output", default="data/airplance_pkl")
# workers
parser.add_argument("--workers", type=int, default=16)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()


output_path = Path(args.output)
output_path.mkdir(parents=True, exist_ok=True)

input_plys = sorted(Path(args.input).glob("*.ply"))

output_pkls = [output_path / (path.stem + '.pkl') for path in input_plys]

tqdm.write(f"Total {len(input_plys)} files")

def get_merge_list(path: Path, output_file: Path):
    if output_file.exists():
        try:
            np.load(output_file)
            return
        except:
            pass
    
    pgs = PGSMoments.load(path, save=False)
    merge_list = pgs.simplify(1)
    merge_list = merge_list[::-1]

    # 建立分裂的连接关系
    tmap = {}
    for merge in merge_list:
        tmap[merge['mixed_id']] = [i if isinstance(i, int) else i.item() for i in [merge['source_id'], merge['target_id']]]
    root = merge_list[0]['mixed_id']
    # 获得分层的GS
    level = 0
    output = {}
    level_step_no_split, level_step_to_split = [], [root]
    while True:
        level_step_split_mask = []
        level_next_no_split, level_next_to_split = [], []

        for index in level_step_to_split:
            if tmap.get(index):
                level_next_to_split.append(tmap[index][0])
                level_next_to_split.append(tmap[index][1])

                level_step_split_mask.append(True)
            else:
                level_next_no_split.append(index)

                level_step_split_mask.append(False)
        
        if len(level_next_to_split) == 0: # 没有分裂的gs
            break

        if len(level_next_to_split) / (len(level_step_no_split)+len(level_step_to_split)) < 0.01:
            break

        level_l_gs = level_step_no_split + level_step_to_split
        output[level] = (
            copy.deepcopy(level_l_gs), 
            copy.deepcopy([False]*len(level_step_no_split)+level_step_split_mask), 
            [tmap[index] for index in level_step_to_split if tmap.get(index)]
        )
        
        level_step_no_split.extend(level_next_no_split)
        level_step_to_split = level_next_to_split

        level += 1
    # 添加GS数据
    output['data'] = pgs._data.copy()
    output['level'] = level # indice: 0~level-1

    with open(output_file, 'wb') as f:
        pickle.dump(output, f)

    tqdm.write(f"{path} done")

def worker(queue, count) -> None:
    while True:
        item = queue.get()
        if item is None:
            break

        get_merge_list(item[0], item[1])
        
        with count.get_lock():
            count.value += 1
        
        queue.task_done()

queue = multiprocessing.JoinableQueue()
count = multiprocessing.Value("i", 0)
processes = []

# use parallel processing
for worker_i in range(args.workers):
    process = multiprocessing.Process(
        target=worker,
        args=(queue, count)
    )
    process.daemon = True
    process.start()
    processes.append(process)

for input_ply, output_npy in zip(input_plys, output_pkls):
    queue.put((input_ply, output_npy))

start_time = time.time()
queue.join()
end_time = time.time()
print(f"All tasks completed in {end_time - start_time:.2f} seconds")
print(f"Processed {count.value} items")

for worker_i in range(args.workers):
    queue.put(None)

for p in processes:
    p.join()
print("All worker processes finished")
