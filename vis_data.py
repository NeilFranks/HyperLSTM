from datasets.hockey         import HockeyDataset
from datasets.hockey_minimal import MinimalHockeyDataset
from plot import *

import torch

def main(*args):
    # full_dataset = HockeyDataset("data/standardized_data.csv")
    full_dataset = MinimalHockeyDataset("data/standardized_data.csv")
    l = len(full_dataset)
    print('Investigating hockey dataset..')
    print(l)
    for i, (x, y) in enumerate(full_dataset):
        if i > 10:
            break
        heatmap(torch.hstack((x, y)), title=f'hockey_{i}', horizontal=True,
                zmin=None, zmax=None)
        print(i, x.shape, y.shape)
    1/0


if __name__ == '__main__':
    main(sys.argv[1:])
