from utils.enhance_data import Augment, enhance_gaussian_field
from pathlib import Path
from tqdm import tqdm, trange
import argparse


def enhance(args):
    augment = Augment()

    # for shapesplat and modelsplat, upaxis = 3
    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_dir():
        if args.type == 'modelsplat':
            upaxis = 3
            pattern = '*/*.ply' if args.glob == '' else args.glob
        elif args.type == 'shapesplat':
            upaxis = 3
            pattern = '*/*.ply' if args.glob == '' else args.glob

        paths = input_path.glob(pattern)

        for path in tqdm(paths):
            target_dir = output_path / path.parent.name
            target_dir.mkdir(exist_ok=True, parents=True)
            for i in trange(args.nums, leave=False):
                target_file = target_dir / f'{path.stem}-enhance-{i}.ply'
                enhance_gaussian_field(path, target_file, augment, upaxis=upaxis)
    else:
        for i in range(args.nums):
            enhance_gaussian_field(args.input, f'{args.output}-enhance-{i}.ply', augment, upaxis=upaxis)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='03636649-cfaf30102d9b7cc6cd6d67789347621.ply')
    parser.add_argument('--output', type=str, default='03636649-enhance.ply')
    parser.add_argument('--nums', type=int, default=10)
    parser.add_argument('--type', type=str, choices=['none', 'modelsplat', 'shapesplat'])
    parser.add_argument('--glob', type=str, default='')
    args = parser.parse_args()

    enhance(args)
