import argparse
import datetime
import random
import time
from pathlib import Path

import torch
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import r2_score

from scipy.stats import pearsonr

from PIL import Image
import cv2
from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
import warnings
from crowd_datasets.SHHA import SHHA
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)
    
    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    parser.add_argument('--output_dir', default='',
                        help='path where to save')
    parser.add_argument('--weight_path', default='',
                        help='path where the trained weights saved')

    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')

    parser.add_argument('--dataset_file', default='SHHA')
    parser.add_argument('--data_root', default='./dataset/dataset_WBC',
                        help='path where the dataset is')
    
    parser.add_argument('--num_workers', default=8, type=int)

    return parser

def average_hausdorff_distance(preds, gts, max_ahd):

    if len(preds) == 0 or len(gts) == 0:
        return max_ahd
    
    preds = np.array(preds)
    gts = np.array(gts)

    assert preds.shape[1] == gts.shape[1]

    d_matrix = pairwise_distances(preds, gts, metric='euclidean')

    return np.average(np.min(d_matrix, axis=0)) + np.average(np.min(d_matrix, axis=1))


def main(args, debug=False):

    CUDA = True
    SAVE = False

    

    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    if not CUDA:
        torch.cuda.is_available = lambda : False
        device = torch.device('cpu')
        print(f'CUDA:{torch.cuda.is_available()}')
    else:
        print(f'CUDA:{torch.cuda.is_available()}')
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        device = torch.device('cuda')
    print(args)
    
    # get the P2PNet
    model = build_model(args)
    # move to GPU
    model.to(device)
    # print(next(model.parameters()).device)
    # load trained model
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    # convert to eval mode
    model.eval()
    # create the pre-processing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    loading_data = build_dataset(args=args)
    train_set, val_set, test_set = loading_data(args.data_root)
    sampler_test = torch.utils.data.SequentialSampler(test_set)

    data_loader_test = DataLoader(test_set, 1, sampler=sampler_test,
                                    drop_last=False, collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers)
    
    
    
    all_preds = []
    all_gts = []

    metrics = {i:[0,0,0] for i in range(1, 16)}

    sum_ahd = 0

    sum_e = 0
    sum_ae = 0
    sum_se = 0
    sum_pe = 0
    sum_ape = 0
    i = 0
    print(len(data_loader_test))
    t1 = time.time()
    for samples, targets in data_loader_test:
        i += 1
        print(f'{i}/{len(data_loader_test)}', end='\r')
       
        img_path = targets[0]['path']
        file_name = targets[0]['path'].split('/')[-1].split('.')[0]

        gts = [t['point'].tolist() for t in targets]
       
        img_raw = Image.open(img_path).convert('RGB')

        samples = samples.to(device)
        # run inference
        outputs = model(samples)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

        outputs_points = outputs['pred_points'][0]
        
        threshold = 0.5
        # filter the predictions
        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()

        predict_cnt = int((outputs_scores > threshold).sum())
        gt_cnt = targets[0]['point'].shape[0]

        all_preds.append(predict_cnt)
        all_gts.append(gt_cnt)

        # print(img_path)

        if len(points) == 0:
            tp = 0
            fp = 0
            fn = len(gts[0])
        else:
            for rad in range(1, 16):
                nbr = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(gts[0])
                dis, idx = nbr.kneighbors(points)
                detected_pts = (dis[:, 0] <= rad).astype(np.uint8)

                nbr = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(points)
                dis, idx = nbr.kneighbors(gts[0])
                detected_gt = (dis[:, 0] <= rad).astype(np.uint8)
                
                tp = np.sum(detected_pts)
                fp = len(points) - tp
                fn = len(gts[0]) - np.sum(detected_gt)
            
                metrics[rad][0] += tp
                metrics[rad][1] += fp
                metrics[rad][2] += fn
                
               
        sum_ahd += average_hausdorff_distance(points,gts[0], math.sqrt(img_raw.size[0]**2 + img_raw.size[1]**2))
        sum_e += predict_cnt - gt_cnt
        sum_ae += abs(predict_cnt - gt_cnt)
        sum_se += (predict_cnt - gt_cnt)**2
        sum_pe += (predict_cnt - gt_cnt) * 100 if gt_cnt == 0 else (predict_cnt -gt_cnt) * 100 / gt_cnt
        sum_ape += abs(predict_cnt - gt_cnt) * 100 if gt_cnt == 0 else abs(predict_cnt - gt_cnt) * 100 / gt_cnt

        # mae = abs(predict_cnt - gt_cnt)
        # mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
        # maes.append(float(mae))
        # mses.append(float(mse))

        # # draw the predictions
        if SAVE:
            size = 2
            img_to_draw_gt = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
            img_to_draw_pred = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
            for p in points:
                img_to_draw_pred = cv2.circle(img_to_draw_pred, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
            
            for t in gts[0]:
                img_to_draw_gt = cv2.circle(img_to_draw_gt, (int(t[0]), int(t[1])), size, (0, 255, 0), -1)
            # save the visualized image
            cv2.imwrite(os.path.join(args.output_dir, file_name+'_pred_{}.jpg'.format(predict_cnt)), img_to_draw_pred)
            cv2.imwrite(os.path.join(args.output_dir, file_name+'_gt_{}.jpg'.format(gt_cnt)), img_to_draw_gt)
        # if i > 200:
        #     break
        

    # mae = np.mean(maes)
    # mse = np.sqrt(np.mean(mses))
    mahd = float(sum_ahd / i)
    me = float(sum_e / i)
    mae = float(sum_ae / i)
    mse = float(sum_se / i)
    mpe = float(sum_pe / i)
    mape = float(sum_ape / i)
    rmse = float(math.sqrt(mse))
    coeff_det = r2_score(all_gts, all_preds)
    person = pearsonr(all_gts, all_preds)[0]

    f_metrics = open('metrics.txt', 'w')
    for r in range(1, 16):
        precision = float(100*metrics[r][0] / (metrics[r][0] + metrics[r][1]))
        recall = float(100*metrics[r][0] / (metrics[r][0] + metrics[r][2]))
        f_metrics.write(f'rad:{r} - precision: {precision}, recall: {recall}, fscore: {float(2 * (precision*recall /(precision+recall)))}\n')
        print(f'rad:{r} - precision: {precision}, recall: {recall}, fscore: {float(2 * (precision*recall /(precision+recall)))}')
    t2 = time.time()
    f_metrics.write(f'ME: {me}, MPE: {mpe}, MAPE: {mape}, MAE: {mae}, MSE: {mse}, RMSE: {rmse}, C_DET: {coeff_det}, PERSON: {person},  MAHD: {mahd}\n')
    print(f'ME: {me}, MPE: {mpe}, MAPE: {mape}, MAE: {mae}, MSE: {mse}, RMSE: {rmse}, C_DET: {coeff_det}, PERSON: {person},  MAHD: {mahd} time: {t2 - t1}')

    f_metrics.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)