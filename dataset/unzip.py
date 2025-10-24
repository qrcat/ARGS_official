import os
import argparse


def upzip_file(input_file, output_dir):
    retrieve = os.system('unzip {} -d {}'.format(input_file, output_dir))
    if retrieve != 0:
        raise RuntimeError('unzip {} failed'.format(input_file))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='')
    parser.add_argument('--output', type=str, default='')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)
    
    if os.path.isdir(args.input):
        for file in os.listdir(args.input):
            if file.endswith('.zip'):
                upzip_file(os.path.join(args.input, file), args.output)
    else:
        upzip_file(args.input, args.output)
