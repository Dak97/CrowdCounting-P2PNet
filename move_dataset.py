from os import listdir, system, rmdir, remove, mkdir, path
import random
from shutil import move, copy2, rmtree

if path.exists('dataset/dataset_WBC/train.list'):
    remove('dataset/dataset_WBC/train.list')
    remove('dataset/dataset_WBC/val.list')
    remove('dataset/dataset_WBC/test.list')

    rmtree('dataset/dataset_WBC/train/scene01/')
    rmtree('dataset/dataset_WBC/val/scene01/')
    rmtree('dataset/dataset_WBC/test/scene01/')

    mkdir('dataset/dataset_WBC/train/scene01/')
    mkdir('dataset/dataset_WBC/val/scene01/')
    mkdir('dataset/dataset_WBC/test/scene01/')

    exit()
# [1,2,3,4,5,6,7,8,9,10,11,12,13,15,17,19,20,30,35,69,76,88,90]
k = 0
for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,15,17,19,20,30,35,69,76,88,90]:
    print(f'CELL: {i}\n')
    f = open(f'dataset/cell/cell_{i}.txt', 'r')
    data = f.read().split('\n')
    data.remove('')
   
    random.shuffle(data)
    num_train = int(len(data) * 0.6)
    num_val = int(len(data) * 0.2)

    train_images = data[0:num_train]
    val_images = data[num_train:num_train+num_val]
    test_images = data[num_train+num_val:]

    # print(len(train_images)+len(val_images)+len(test_images))

    f_train = open('dataset/dataset_WBC/train.list', 'a')
    f_val = open('dataset/dataset_WBC/val.list', 'a')
    f_test = open('dataset/dataset_WBC/test.list', 'a')

    
    if i > 9:
        for im,i in zip(data,range(len(data))):
            
            if k % 3 == 0:
                print(f'Train: {im}')
                copy2(f'dataset/images_cropped/{im}.jpg', 'dataset/dataset_WBC/train/scene01')
                copy2(f'dataset/images_cropped/{im}.txt', 'dataset/dataset_WBC/train/scene01')
                f_train.write(f'train/scene01/{im}.jpg train/scene01/{im}.txt\n')
            if k % 3 == 1:
                print(f'Val: {im}')
                copy2(f'dataset/images_cropped/{im}.jpg', 'dataset/dataset_WBC/val/scene01')
                copy2(f'dataset/images_cropped/{im}.txt', 'dataset/dataset_WBC/val/scene01')
                f_val.write(f'val/scene01/{im}.jpg val/scene01/{im}.txt\n')
            if k % 3 == 2:
                print(f'Test: {im}')
                copy2(f'dataset/images_cropped/{im}.jpg', 'dataset/dataset_WBC/test/scene01')
                copy2(f'dataset/images_cropped/{im}.txt', 'dataset/dataset_WBC/test/scene01')
                f_test.write(f'test/scene01/{im}.jpg test/scene01/{im}.txt\n')
            k+=1
    else:
        i = 0
        for im in train_images:
            i = i + 1
        
            print(f'Train: {i}/{len(train_images)}')
            copy2(f'dataset/images_cropped/{im}.jpg', 'dataset/dataset_WBC/train/scene01')
            copy2(f'dataset/images_cropped/{im}.txt', 'dataset/dataset_WBC/train/scene01')
            f_train.write(f'train/scene01/{im}.jpg train/scene01/{im}.txt\n')

        i = 0
        for im in val_images:
            i = i + 1
            
            print(f'Val: {i}/{len(val_images)}')
            copy2(f'dataset/images_cropped/{im}.jpg', 'dataset/dataset_WBC/val/scene01')
            copy2(f'dataset/images_cropped/{im}.txt', 'dataset/dataset_WBC/val/scene01')
            f_val.write(f'val/scene01/{im}.jpg val/scene01/{im}.txt\n')
        i = 0
        for im in test_images:
            i = i + 1
            
            print(f'Test: {i}/{len(test_images)}')
            copy2(f'dataset/images_cropped/{im}.jpg', 'dataset/dataset_WBC/test/scene01')
            copy2(f'dataset/images_cropped/{im}.txt', 'dataset/dataset_WBC/test/scene01')
            f_test.write(f'test/scene01/{im}.jpg test/scene01/{im}.txt\n')

f.close()
f_train.close()
f_val.close()
f_test.close()
