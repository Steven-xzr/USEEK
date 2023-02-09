# -*- coding: utf-8 -*-
import numpy as np
import argparse
import json


arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('-p', '--prediction', type=str, default='./prediction/chair_segment_d01_ns3_nms02.npz',
                        help='Prediction file from predictor output.')
arg_parser.add_argument('--out-path', type=str, default='./vis_chair.json')
args = arg_parser.parse_args()


if __name__ == '__main__':
    predicted = np.load(args.prediction)
    preds = predicted['pred']    # D x K x 3
    pcds = predicted['pcd']      # D x N x 3

    out = []
    for pred, pcd in zip(preds, pcds):
        data = {'keypoint': pred.tolist(),
                'original': pcd.tolist(),
                }
        out.append(data)

    with open(args.out_path, 'w') as f:
        json.dump(out[:100], f)
