from utils.io import activated_gs2train_gs
from pgs import PGS, PGSMoments
import numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing
import argparse
import time

parser = argparse.ArgumentParser()
# inputs
parser.add_argument("--shapesplat_ply", default="/mnt/private_rqy/gs_data/shapesplat_ply")
parser.add_argument("--modelsplat_ply", default="/mnt/private_rqy/gs_data/modelsplat_ply")
# outputs
parser.add_argument("--output", default="/mnt/private_rqy/gs_data/merge")
# workers
parser.add_argument("--workers", type=int, default=16)
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

def get_merge_list(path: Path, output: Path):
    if output.exists():
        try:
            np.load(output)
            return
        except:
            pass
    
    pgs = PGSMoments.load(path, save=False)
    merge_list = pgs.simplify(1)
    merge_list = merge_list[::-1]
    source = np.stack([np.stack((i['source'], i['target']), axis=1) for i in merge_list])
    target = np.stack([i['mixed'] for i in merge_list])

    np.savez(output, source=source, target=target)

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

# pgs = PGSMoments.load("gradio_output.ply")
# for size in [32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]:
#     pgs.simplify(size)
#     pgs.save(f"output/zip-{size}.ply")


# pgs = PGS.load("gradio_output.ply")
# for size in [32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]:
#     pgs.simplify(size)
#     pgs.save(f"zip-{size}.ply")
# merge_list = pgs.simplify(1)
# merge_list = merge_list[::-1]

# source = np.stack([np.stack((i['source'], i['target']), axis=1) for i in merge_list])
# target = np.stack([i['mixed'] for i in merge_list])

# np.savez("datasets/merge_origin/0.npz", source=source, target=target)

# source = source.transpose(0, 2, 1)
# source = activated_gs2train_gs(source)
# target = activated_gs2train_gs(target)

# np.savez("datasets/merge/0.npz", source=source, target=target)
