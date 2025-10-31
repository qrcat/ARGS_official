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


def get_merge_list(data_index:int, path: Path, output_file: Path, text=''):
    # if output_file.exists():
    #     try:
    #         with open(output_file, 'rb') as f:
    #             pickle.load(f)
    #         return
    #     except:
    #         pass
    
    pgs = PGSMoments.load(path, save=False)
    merge_list = pgs.simplify(1, 'merge_gaussian_moments_ub')
    merge_list = merge_list[::-1]

    # 建立分裂的连接关系
    tmap = {}
    for merge in merge_list:
        tmap[merge['mixed_id']] = [i if isinstance(i, int) else i.item() for i in [merge['source_id'], merge['target_id']]]
    root = merge_list[0]['mixed_id']
    # 获得分层的GS
    prev_gs_to_split = [root]
    count = []
    sequence, split_gs, split_bl = [], [], []
    output = {}
    while True:
        next_gs_to_split = []

        count.append(len(prev_gs_to_split))

        for index in prev_gs_to_split:
            sequence.append(index)
            if tmap.get(index):
                split_gs.append([tmap[index][0], tmap[index][1]])
                split_bl.append(True)

                next_gs_to_split.append(tmap[index][0])
                next_gs_to_split.append(tmap[index][1])
            else:
                split_gs.append([index, index]) # 不分裂，填充自己的特征
                split_bl.append(False)
        
        if len(next_gs_to_split) == 0: # 没有分裂的gs
            break

        prev_gs_to_split = next_gs_to_split

    cumsum = np.cumsum([0]+count[:-1])

    output['data'] = pgs._data
    output['count'] = np.array(count)
    output['cumsum'] = cumsum
    output['sequence'] = np.array(sequence)
    output['split_gs'] = np.array(split_gs)
    output['split_bl'] = np.array(split_bl)

    with open(output_file, 'wb') as f:
        pickle.dump(output, f)

    tqdm.write(f"{data_index}: {path} done")

def worker(queue, count) -> None:
    while True:
        item = queue.get()
        if item is None:
            break

        get_merge_list(*item)
        
        with count.get_lock():
            count.value += 1
        
        queue.task_done()


def main(args):
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.type == 'modelsplat':
        pattern = '*/point_cloud.ply'
    
    input_plys = sorted(Path(args.input).glob(pattern))

    tqdm.write(f"Total {len(input_plys)} files")

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

    for i, input_ply in enumerate(input_plys):
        target_dir = output_path / input_ply.parent.name
        target_dir.mkdir(exist_ok=True, parents=True)
        output_pkl = target_dir / (input_ply.stem + '_block.pkl')

        if args.type == 'modelsplat':
            text = input_ply.parent.name.split('_')[0]

        queue.put((i, input_ply, output_pkl, text))

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # inputs
    parser.add_argument("--input", default="data/airplane_enhance")
    # outputs
    parser.add_argument("--output", default="data/airplance_pkl")
    # type
    parser.add_argument("--type", default="none", choices=['none', 'modelsplat', 'shapesplat'])
    # workers
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(args)
