import argparse

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from train.hash_train import DDSPHTrainer

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision('high')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="coco", help="choice from [coco, flickr, nuswide, iapr]")
    parser.add_argument("--output-dim", type=int, default=16)
    parser.add_argument("--is-train", default=True)

    parser.add_argument("-backbone", type=str, default="clip")
    parser.add_argument("-preload", type=str, default="/home/yuebai/Data/Preload/ViT-B-32.pt")
    
    args = parser.parse_args()

    DDSPHTrainer(args, 0)


if __name__ == "__main__":
    main()

