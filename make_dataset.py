from os import listdir, system
from PIL import Image, ImageDraw, ImageOps
import cv2
import numpy as np
import math

def get_center_bb(coord, x_orig, y_orig, x_max, y_max, scale_percent):

    x_co = [coord[0], coord[2]]
    y_co = [coord[1], coord[3]]
    p_1 = [p - x_orig for p in x_co]
    p_2 = [p - y_orig for p in y_co]

    p1 = [p_1[0], p_2[0]]
    p2 = [p_1[1], p_2[1]]

    p3 = [p2[0], p1[1]]
    p4 = [p1[0], p2[1]]

    x_ce = int ((p1[0] + p3[0]) / 2)
    y_ce = int ((p1[1] + p4[1]) / 2)

    x_ce = int(x_ce * scale_percent / 100)
    y_ce = int(y_ce * scale_percent / 100)
    if x_ce > 0 and x_ce < x_max and y_ce > 0 and y_ce < y_max:
        return x_ce, y_ce
    else:
        return -1,-1
    

def cropping(img):
    # read image
    # img = cv2.imread('img.jpg')
    h, w = img.shape[:2]

    # threshold so border is black and rest is white (invert as needed). 
    # Here I needed to specify the upper threshold at 20 as your black is not pure black.

    lower = (0,0,0)
    upper = (20,20,20)
    mask = cv2.inRange(img, lower, upper)
    mask = 255 - mask

    # define top and left starting coordinates and starting width and height
    top = 0
    left = 0
    bottom = h
    right = w

    # compute the mean of each side of the image and its stop test
    mean_top = np.mean( mask[top:top+1, left:right] )
    mean_left = np.mean( mask[top:bottom, left:left+1] )
    mean_bottom = np.mean( mask[bottom-1:bottom, left:right] )
    mean_right = np.mean( mask[top:bottom, right-1:right] )

    mean_minimum = min(mean_top, mean_left, mean_bottom, mean_right)

    top_test = "stop" if (mean_top == 255) else "go"
    left_test = "stop" if (mean_left == 255) else "go"
    bottom_test = "stop" if (mean_bottom == 255) else "go"
    right_test = "stop" if (mean_right == 255) else "go"

    # iterate to compute new side coordinates if mean of given side is not 255 (all white) and it is the current darkest side
    while top_test == "go" or left_test == "go" or right_test == "go" or bottom_test == "go":

        # top processing
        if top_test == "go":
            if mean_top != 255:
                if mean_top == mean_minimum:
                    top += 1
                    mean_top = np.mean( mask[top:top+1, left:right] )
                    mean_left = np.mean( mask[top:bottom, left:left+1] )
                    mean_bottom = np.mean( mask[bottom-1:bottom, left:right] )
                    mean_right = np.mean( mask[top:bottom, right-1:right] )
                    mean_minimum = min(mean_top, mean_left, mean_right, mean_bottom)
                    #print("top",mean_top)
                    continue
            else:
                top_test = "stop"   

        # left processing
        if left_test == "go":
            if mean_left != 255:
                if mean_left == mean_minimum:
                    left += 1
                    mean_top = np.mean( mask[top:top+1, left:right] )
                    mean_left = np.mean( mask[top:bottom, left:left+1] )
                    mean_bottom = np.mean( mask[bottom-1:bottom, left:right] )
                    mean_right = np.mean( mask[top:bottom, right-1:right] )
                    mean_minimum = min(mean_top, mean_left, mean_right, mean_bottom)
                    #print("left",mean_left)
                    continue
            else:
                left_test = "stop"  

        # bottom processing
        if bottom_test == "go":
            if mean_bottom != 255:
                if mean_bottom == mean_minimum:
                    bottom -= 1
                    mean_top = np.mean( mask[top:top+1, left:right] )
                    mean_left = np.mean( mask[top:bottom, left:left+1] )
                    mean_bottom = np.mean( mask[bottom-1:bottom, left:right] )
                    mean_right = np.mean( mask[top:bottom, right-1:right] )
                    mean_minimum = min(mean_top, mean_left, mean_right, mean_bottom)
                    #print("bottom",mean_bottom)
                    continue
            else:
                bottom_test = "stop"    

        # right processing
        if right_test == "go":
            if mean_right != 255:
                if mean_right == mean_minimum:
                    right -= 1
                    mean_top = np.mean( mask[top:top+1, left:right] )
                    mean_left = np.mean( mask[top:bottom, left:left+1] )
                    mean_bottom = np.mean( mask[bottom-1:bottom, left:right] )
                    mean_right = np.mean( mask[top:bottom, right-1:right] )
                    mean_minimum = min(mean_top, mean_left, mean_right, mean_bottom)
                    #print("right",mean_right)
                    continue
            else:
                right_test = "stop" 


    # crop input
    result = img[top:bottom, left:right]

    # print crop values 
    # print("top: ",top)
    # print("bottom: ",bottom)
    # print("left: ",left)
    # print("right: ",right)
    # print("height:",result.shape[0])
    # print("width:",result.shape[1])  

    return mask, result, left, top, result.shape[0], result.shape[1]
    # save cropped image
    
    # cv2.imwrite('img_cropped.png',result)
    # cv2.imwrite('img_mask.png',mask)

    # # show the images
    # cv2.imshow("mask", mask)
    # cv2.imshow("cropped", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

files = listdir('dataset/images')

# files = ['20160720_232843.jpg']
ind = 0
m = len(files)

for file in files:
    ind = ind + 1

    print(f'{ind}/{m} - {file}', end='\r')
    
    f_log =  open(f'log.txt', 'a')
    # im = Image.open(f'dataset/images/{file}')
    im = cv2.imread(f'dataset/images/{file}')
    # im = ImageOps.exif_transpose(im)

    file_json = file.split('.')[0]
    
    f = open(f'dataset/jsons/{file_json}.json', 'r')

    data = f.read()
    if data.find('null'):
        data = data.replace('null', 'None')
    data = eval(data)
    f.close()

    file_txt = open(f'dataset/images_cropped/{file_json}.txt', 'w')
    # file_txt = open(f'{file_json}.txt', 'w')
    num_cell = data['Cell Numbers']
    # print(f'Numero di cellule: {num_cell}')
    cell_coord = []
    for i in range(num_cell):
        coord = (int(data[f'Cell_{i}']['x1']), int(data[f'Cell_{i}']['y1']), int(data[f'Cell_{i}']['x2']), int(data[f'Cell_{i}']['y2']))
        cell_coord.append(coord)

    all_coord = []
    for coord in cell_coord:
        all_coord = all_coord + list(coord)

    im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)

    mask, im_crop, x1, y1, y_max, x_max = cropping(im)
    
    # img_crop = cv2.circle(img_crop, (2650,1091), radius=10, color=(0, 0, 255), thickness=-1)
    # img_crop = cv2.circle(img_crop, (3225,1666), radius=10, color=(0, 0, 255), thickness=-1)
   
    scale_percent = 30
    width = int(im_crop.shape[1] * scale_percent / 100)
    height = int(im_crop.shape[0] * scale_percent / 100)

    dim = (width, height)

    im_res = cv2.resize(im_crop, dim)

    cv2.imwrite(f'dataset/images_cropped/{file}', im_res)
   
    for coord in cell_coord:
        x, y = get_center_bb(coord, x1, y1, im_res.shape[1], im_res.shape[0], scale_percent)
        if x != -1 and y != -1:
            
            file_txt.write(f'{x} {y}\n')
        else:
            f_log.write(f'File name: {file}, Cell index: {cell_coord.index(coord)}, Num_cell: {num_cell}\n')
            print(f'File name: {file}, Cell index: {cell_coord.index(coord)}, Num_cell: {num_cell}')
    
    file_txt.close()
    f_log.close()
    



