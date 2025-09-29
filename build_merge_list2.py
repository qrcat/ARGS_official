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
parser.add_argument("--shapesplat_ply", default="/mnt/private_rqy/gs_data/shapesplat_ply")
parser.add_argument("--modelsplat_ply", default="/mnt/private_rqy/gs_data/modelsplat_ply")
# outputs
parser.add_argument("--output", default="/apdcephfs/share_303772734/rqy/gs_merge/")
# workers
parser.add_argument("--workers", type=int, default=32)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()


output_path = Path(args.output)
output_path.mkdir(parents=True, exist_ok=True)

shapesplat_plys = sorted(Path(args.shapesplat_ply).glob("*.ply"))
modelsplat_plys = sorted(Path(args.modelsplat_ply).glob("*/*/*/*.ply"))
shapesplat_npys = [output_path / (path.stem + '.npz') for path in shapesplat_plys]
modelsplat_npys = [
    output_path / ('_'.join(path.as_posix().split('/')[-4:-1]) + '.npz') for path in modelsplat_plys
]

input_plys = shapesplat_plys + modelsplat_plys
output_npys = shapesplat_npys + modelsplat_npys

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
        tmap[merge['mixed_id']] = [merge['source_id'], merge['target_id']]
    root = merge_list[0]['mixed_id']
    # 获得分层的GS
    level = 0
    output = {}
    now_gs_index = [root]
    while True:
        next_gs_index = []
        next_gs_split = []
        for index in now_gs_index:
            if tmap.get(index):
                next_gs_index.append(tmap[index][0])
                next_gs_index.append(tmap[index][1])
                next_gs_split.append(True)
            else:
                next_gs_index.append(index)
                next_gs_split.append(False)
        if sum(next_gs_split) == 0: # 没有分裂的gs
            break
        output[level] = (now_gs_index, next_gs_split, [tmap[index] for index in now_gs_index if tmap.get(index)])
        now_gs_index = copy.deepcopy(next_gs_index)

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

for input_ply, output_npy in zip(input_plys, output_npys):
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
