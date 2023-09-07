from os import listdir, system
from PIL import Image, ImageDraw, ImageOps

def get_center_bb(coord, x_orig, y_orig):

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

    return x_ce, y_ce

files = listdir('dataset/images')

# files = ['20160721_010054.jpg']
ind = 0
m = len(files)

for file in files:
    ind = ind + 1

    print(f'{ind}/{m} - {file}')
    

    im = Image.open(f'dataset/images/{file}')
    # im = ImageOps.exif_transpose(im)

    file_json = file.split('.')[0]
    
    f = open(f'dataset/jsons/{file_json}.json', 'r')

    data = f.read()
    if data.find('null'):
        data = data.replace('null', 'None')
    data = eval(data)
    f.close()

    file_txt = open(f'dataset/images_cropped/{file_json}.txt', 'w')

    num_cell = data['Cell Numbers']
    # print(f'Numero di cellule: {num_cell}')
    cell_coord = []
    for i in range(num_cell):
        coord = (int(data[f'Cell_{i}']['x1']), int(data[f'Cell_{i}']['y1']), int(data[f'Cell_{i}']['x2']), int(data[f'Cell_{i}']['y2']))
        cell_coord.append(coord)

    # im_draw = ImageDraw.Draw(im)
    # for coord in cell_coord:
    #     p1 = [coord[0], coord[1]]
    #     p2 = [coord[2], coord[3]]
        
    #     x_1 = p1[0] 
    #     y_1 = p1[1]

    #     x_2 = p2[0] 
    #     y_2 = p2[1]

    #     print(x_1, x_2, y_1, y_2)
    #     im_draw.ellipse((x_1, y_1, x_2, y_2),  outline ='red')

    # im.show()
    # system('pause')
    # print('ciao')

    all_coord = []
    for coord in cell_coord:
        all_coord = all_coord + list(coord)

    x_coord = tuple(all_coord[::2])
    y_coord = tuple(all_coord[1::2])
   
    x_min = min(x_coord)
    x_max = max(x_coord)
    
    y_min = min(y_coord)
    y_max = max(y_coord)

    deltaH = 0
    deltaW = 0
    x1 = x_min - deltaW
    y1 = y_min - deltaH
    x2 = x_max + deltaW
    y2 = y_max + deltaH 
    # print(x1, y1, x2, y2)
    # print(im.size)
    crop = (x1, y1, x2, y2)
    im_crop = im.crop(crop)

    # im.show()

    im_crop.save(f'dataset/images_cropped/{file}')
    # save image cropped

    for coord in cell_coord:
        x, y = get_center_bb(coord, x1, y1)
        # print(x, y)
        file_txt.write(f'{x} {y}\n')
    
    file_txt.close()
    



