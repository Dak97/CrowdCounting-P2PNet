import argparse
import datetime
import random
import time
from pathlib import Path

import torch
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import numpy as np

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

def main(args, debug=False):

    CUDA = True
    SAVE = True

    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    if not CUDA:
        torch.cuda.is_available = lambda : False
        device = torch.device('cpu')
        print(torch.cuda.is_available())
    else:
        print(torch.cuda.is_available())
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
    
    
        

    
    # names = os.listdir('dataset/dataset_WBC/test/scene01')[:4]

    # names = list(filter(lambda x: x.split('.')[1]=='jpg', names))
    maes = []
    mses = []
    i = 0
    print(len(data_loader_test))
    t1 = time.time()
    for samples, targets in data_loader_test:
        i += 1
        print(f'{i}/{len(data_loader_test)}', end='\r')
        # print(targets[0])
        # pre_im = str(targets[0]['pre_image_id'].item())
        # name_im = str(targets[0]['image_id'].item())
        # if len(name_im) < 6:
        #     name_im = '00'+name_im
        # file_name = f'{pre_im}_{name_im}.jpg'

        img_path = targets[0]['path']
        file_name = targets[0]['path'].split('/')[-1].split('.')[0]

        gts = [t['point'].tolist() for t in targets]
        # # set your image path here
        # # file_name = '20160720_232423'
        # img_path = f'dataset/dataset_WBC/test/scene01/{file_name}'
        # # load the images
        img_raw = Image.open(img_path).convert('RGB')
        # # round the size
        
        # width, height = img_raw.size
        # new_width = width // 128 * 128
        # new_height = height // 128 * 128
        # img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
        # # pre-proccessing
        # img = transform(img_raw)

        # samples = torch.Tensor(img).unsqueeze(0)
        # # samples = samples.to('cuda')
        # # print(samples.get_device())
        samples = samples.to(device)
        # run inference
        outputs = model(samples)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

        outputs_points = outputs['pred_points'][0]
        
        threshold = 0.5
        # filter the predictions
        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
        predict_cnt = int((outputs_scores > threshold).sum())

        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

        outputs_points = outputs['pred_points'][0]
        
        gt_cnt = targets[0]['point'].shape[0]

        mae = abs(predict_cnt - gt_cnt)
        mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
        maes.append(float(mae))
        mses.append(float(mse))
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
            # if i > 10:
            #     break
        

    mae = np.mean(maes)
    mse = np.sqrt(np.mean(mses))
    t2 = time.time()
    print(f'mae: {mae} mse: {mse} time: {t2 - t1}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)