import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy
import json

# img=mpimg.imread(r'E:\PyProjects\CN4118_FYP\CNNtest\data\Classification\single_010.jpg')
# img= img[:,:,0]
#
# x, y = img.shape
#
# imgplot = plt.imshow(img, cmap='gray')
# plt.show()
#
# single= numpy.zeros([100, 100])
# center= numpy.array([x//2, y//2+3])
# r= 37
#
# for i in range(x):
#     for j in range(y):
#         d2 = (i-center[0])**2 + (j-center[1])**2
#         if d2 < r**2:
#             single[50+i-center[0],50+j-center[1]]=img[i,j]
#
# single= single[(50-r):(50+r), (50-r):(50+r)]
# imgplot = plt.imshow(single, cmap='gray')
# plt.show()
#
# mpimg.imsave('E:\PyProjects\CN4118_FYP\Database\Single_R\R002.jpg', single,cmap='gray')

img = mpimg.imread(r'E:\PyProjects\CN4118_FYP\Database\Single_R\R002.jpg')
pathL = r'E:\PyProjects\CN4118_FYP\Database\Single_L'
pathR = r'E:\PyProjects\CN4118_FYP\Database\Single_R'
c = 0
dataset = [None] * 4
for L in range(2):
    path2 = r'\L' + str(L + 1).rjust(3, '0') + '.jpg'
    img = mpimg.imread(pathL + path2)
    img = img[:, :, 0]
    dataset[c] = [1, img.tolist()]

    c = c + 1

for R in range(2):
    path2 = r'\R' + str(R + 1).rjust(3, '0') + '.jpg'
    img = mpimg.imread(pathR + path2)
    img = img[:, :, 0]
    dataset[c] = [2, img.tolist()]

    c = c + 1

with open('E:\PyProjects\CN4118_FYP\Database\dataset.txt', 'w') as out_file:
    json.dump(dataset, out_file)