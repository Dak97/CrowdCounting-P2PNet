from matplotlib import image, transforms

from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageOps

PLOT_CROP = True
file_image = '20160721_022014.jpg' 
file_json = file_image.split('.')[0]
points = []

if PLOT_CROP:
    im = image.imread(f'dataset/images_cropped/{file_image}')
    f = open(f'dataset/images_cropped/{file_json}.txt', 'r')
    data = f.read()
    f.close()

    data = data.split('\n')
    points = [int(item) for row in [d.split(' ') for d in data] for item in row if item is not '']
    l = len(points)-1
    points = [(points[i], points[i+1]) for i in range(0,(len(points)-1),2)]

# im_or = image.imread(f'dataset/images/{file_image}')
# # im_crop = Image.open(f'images/{file_image}')

# f_json = open(f'dataset/jsons/{file_json}.json', 'r')
# data_json = eval(f_json.read())

# f_json.close()


# num_cell = data_json['Cell Numbers']
# print(f'Numero di cellule: {num_cell}')
# cell_coord = []
# for i in range(num_cell):
#     coord = (int(data_json[f'Cell_{i}']['x1']), int(data_json[f'Cell_{i}']['y1']), int(data_json[f'Cell_{i}']['x2']), int(data_json[f'Cell_{i}']['y2']))
#     cell_coord.append(coord)

fig, (ax1, ax2) = plt.subplots(1, 2)


# for cell in cell_coord:
#     p1 = [cell[0], cell[1]]
#     p2 = [cell[2], cell[3]]

#     p3 = [p2[0], p1[1]]
#     p4 = [p1[0], p2[1]]

#     x1 = [p1[0], p3[0]]
#     y1 = [p1[1], p3[1]]
#     x2 = [p3[0], p2[0]]
#     y2 = [p3[1], p2[1]]
#     x3 = [p2[0], p4[0]]
#     y3 = [p2[1], p4[1]]
#     x4 = [p4[0], p1[0]]
#     y4 = [p4[1], p1[1]]

#     print(p1, p2, im_or.shape)
    


#     xc = int ((p1[0] + p3[0]) / 2)
#     yc = int ((p1[1] + p4[1]) / 2)

#     ax1.plot(xc, yc, marker='.', color='white')
#     ax1.plot(x1, y1,x2, y2,x3, y3, x4, y4, marker='.', color='red')

# ax1.imshow(im_or)
# plt.show()
for cell in points:
    # p1 = [cell[0], cell[1]]
    # p2 = [cell[2], cell[3]]

    # p3 = [p2[0], p1[1]]
    # p4 = [p1[0], p2[1]]

    # x1 = [p1[0], p3[0]]
    # y1 = [p1[1], p3[1]]
    # x2 = [p3[0], p2[0]]
    # y2 = [p3[1], p2[1]]
    # x3 = [p2[0], p4[0]]
    # y3 = [p2[1], p4[1]]
    # x4 = [p4[0], p1[0]]
    # y4 = [p4[1], p1[1]]


    # xc = int ((p1[0] + p3[0]) / 2)
    # yc = int ((p1[1] + p4[1]) / 2)

    ax2.plot(cell[0], cell[1], marker='.', color='red')
    # plt.plot(x1, y1,x2, y2,x3, y3, x4, y4, marker='.', color='red')

if PLOT_CROP:
    ax2.imshow(im)
plt.show()


# all_coord = ()
# for coord in cell_coord:
#     all_coord = all_coord + coord

# x_coord = tuple(filter(lambda x: all_coord.index(x) % 2 == 0, all_coord))
# y_coord = tuple(filter(lambda x: all_coord.index(x) % 2 == 1, all_coord))

# x_min = min(x_coord)
# x_max = max(x_coord)

# y_min = min(y_coord)
# y_max = max(y_coord)

# deltaH = 100
# deltaW = 300
# x1 = x_min - deltaW
# y1 = y_min - deltaH
# x2 = x_max + deltaW
# y2 = y_max + deltaH 
# crop = (x1, y1, x2, y2)
# im_crop1 = im_crop.crop(crop)
# print(f'w: {im_crop1.width}\nh: {im_crop1.height}')

# im_draw = ImageDraw.Draw(im_crop1)
# print(x1, x2, y1, y2)

# for coord in cell_coord:
#     p1 = [coord[0], coord[1]]
#     p2 = [coord[2], coord[3]]
#     print(p1)
#     print(p2)
#     x_1 = p1[0] - x1
#     y_1 = p1[1] - y1

#     x_2 = p2[0] - x1
#     y_2 = p2[1] - y1

#     print(x_1, x_2, y_1, y_2)
#     im_draw.ellipse((x_1, y_1, x_2, y_2),  outline ='blue')

# im_crop1.show()


    
    