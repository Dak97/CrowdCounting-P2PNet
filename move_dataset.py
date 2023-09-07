from os import listdir
import random
from shutil import move

files = listdir('dataset/images_cropped/')
names = [f.split('.')[0]for f in files]

# random.shuffle(names)
# num_train = int(len(names) * 0.8)
# num_val = int(len(names) * 0.1)

# train_images = names[0:num_train]
# val_images = names[num_train:num_train+num_val]

# f_train = open('dataset/dataset_WBC/train.list', 'w')
# f_val = open('dataset/dataset_WBC/test.list', 'w')
f_test = open('dataset/dataset_WBC/test1.list', 'w')
# i = 0
# for im in train_images:
#     i = i + 1
#     print(f'Train: {i}/{len(train_images)}')
#     move(f'dataset/images_cropped/{im}.jpg', 'dataset/dataset_WBC/train/scene01')
#     move(f'dataset/images_cropped/{im}.txt', 'dataset/dataset_WBC/train/scene01')
#     f_train.write(f'train/scene01/{im}.jpg train/scene01/{im}.txt\n')
# i = 0
# for im in val_images:
#     i = i + 1
#     print(f'Test: {i}/{len(val_images)}')
#     move(f'dataset/images_cropped/{im}.jpg', 'dataset/dataset_WBC/test/scene01')
#     move(f'dataset/images_cropped/{im}.txt', 'dataset/dataset_WBC/test/scene01')
#     f_val.write(f'test/scene01/{im}.jpg test/scene01/{im}.txt\n')
i = 0
for im in list(set(names)):
    i = i + 1
    print(f'Test: {i}/{len(list(set(names)))}')
    move(f'dataset/images_cropped/{im}.jpg', 'dataset/dataset_WBC/test1/scene01')
    move(f'dataset/images_cropped/{im}.txt', 'dataset/dataset_WBC/test1/scene01')
    f_test.write(f'test1/scene01/{im}.jpg test1/scene01/{im}.txt\n')