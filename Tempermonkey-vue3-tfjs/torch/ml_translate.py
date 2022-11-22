import argparse
import sys


def get_args():
    parser = argparse.ArgumentParser(description='simple ml')
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    parser.add_argument("-p", "--prepare", help="optional prepare data", dest="prepare", action='store_true')
    parser.add_argument("-t", "--train", help="optional train data", dest="train", action='store_true')
    parser.add_argument("-e", "--test", help="optional test", dest="test", action='store_true')
    return parser.parse_args()


def prepare_data():
    print(f"start preparing data")


def train():
    print(f"start training")


def test():
    print(f"test")


if __name__ == '__main__':
    args = get_args()
    if args.train:
        train()
        sys.exit(0)
    if args.prepare:
        prepare_data()
        sys.exit(0)
    if args.test:
        test()
        sys.exit(0)
    

