from os import listdir, system, remove

files = listdir('dataset/images_cropped')

files = list(filter(lambda txt: txt.split('.')[1] == 'txt', files))

ind = 0
m = len(files)

tot = 0
num_cell = 0
acc = 0
for i in range(130):
    f_cell = open(f'dataset/cell/cell_{i}.txt', 'w')

    for file_name in files:

        ind = ind + 1

        # print(f'{ind}/{m} - {file_name}' , end='\r')

        file_name = file_name.split('.')[0]

        f = open(f'dataset/images_cropped/{file_name}.txt', 'r')
        num = len(f.readlines())
        f.close()
        
        if num == num_cell:
            tot = tot + 1
            acc += 1
            # remove(f'dataset/images_cropped/{file_name}.txt')
            # remove(f'dataset/images_cropped/{file_name}.jpg')
            
            f_cell.write(f'{file_name}\n')

    f_cell.close()

    if tot == 0:
        remove(f'dataset/cell/cell_{i}.txt')

    if tot != 0:
        print(f'num_cell: {num_cell}, tot:{tot} <-------')
    else:
        print(f'num_cell: {num_cell}, tot:{tot}')
    tot = 0
    num_cell += 1

print(f'acc: {acc}')