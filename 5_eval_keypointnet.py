# -*- coding: utf-8 -*-
import numpy as np
import argparse


arg_parser = argparse.ArgumentParser(description="Evaluation for detected keypoints.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
arg_parser.add_argument('-a', '--annotation-json', type=str, default='./data/keypointnet/annotations/knife.json',
                        help='Annotation JSON file path from KeypointNet dataset.')
arg_parser.add_argument('-i', '--pcd-path', type=str, default='./data/keypointnet/pcds',
                        help='Point cloud file folder path from KeypointNet dataset.')
arg_parser.add_argument('-p', '--prediction', type=str, default='./data/prediction/knife_sprin.npz',
                        help='Prediction file from predictor output.')
arg_parser.add_argument('--miou-threshold', type=float, default=0.2)
args = arg_parser.parse_args()


def alignment_scores(keypoints1, keypoints2):
    preds = []
    for kp1, kp2 in zip(keypoints1, keypoints2):
        kp1_e = np.expand_dims(kp1, 1)  # k1 x 1 x 3
        kp2_e = np.expand_dims(kp2, 0)  # 1 x k2 x 3
        dist = np.sum(np.square(kp1_e - kp2_e), -1)  # k1 x k2
        idx = np.argmin(dist, -1)  # k1
        preds.append(idx)
    preds = np.array(preds, dtype=np.int32)  # n x k1
    acc = []
    for pa in preds:
        for pb in preds:
            acc.append(np.mean(pa == pb))
    return np.mean(acc)


def mIoU(prediction, ground_truth, threshold=0.1):
    npos = 0
    fp_sum = 0
    fn_sum = 0
    for ground_truths, kpcd in zip(ground_truth, prediction):
        kpcd_e = np.expand_dims(kpcd, 1)  # k1 x 1 x 3
        gt_e = np.expand_dims(ground_truths, 0)  # 1 x k2 x 3
        dist = np.sqrt(np.sum(np.square(kpcd_e - gt_e), -1))  # k1 x k2
        npos += len(np.min(dist, -2))
        fp_sum += np.count_nonzero(np.min(dist, -1) > threshold)
        fn_sum += np.count_nonzero(np.min(dist, -2) > threshold)
    return (npos - fn_sum) / (npos + fp_sum)


if __name__ == '__main__':
    predicted = np.load(args.prediction)
    pred = predicted['pred']
    gt = predicted['gt']
    print("mIoU:", mIoU(pred, gt, args.miou_threshold))
    print("Dual Alignment Score:", (alignment_scores(pred, gt) + alignment_scores(gt, pred)) / 2)
