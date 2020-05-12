import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import imutils
import numpy
import pandas

def myrotate(img,deg):
    r = imutils.rotate_bound(img, deg)
    return r

def myreshape(im, desired_size=90):
    old_size = im.shape[:2]
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = 0
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)
    return new_im

def mylum(img, lum):
    img= img*lum
    img= numpy.clip(img, 0, 1)
    return img

# data = pandas.read_csv('E:\PyProjects\data\Classification\Label.txt', sep=" ", header=None)
# L= data[0].size
# Dataset= numpy.empty([L*25,90*90+1])
# Count=0
#
# Path1= r'E:\PyProjects\data\Classification'+"\\"
# PathL='E:\PyProjects\data\Classification\L'+"\\"
# PathR='E:\PyProjects\data\Classification\R'+"\\"
#
# for i in range(0,58,1):
#   Path2= data[0][i]
#   Label1= data[1][i]
#
#   img1 = mpimg.imread(Path1+Path2)
#   img1 = img1[:, :, 0] / 255
#
#   for deg in (-30, -15, 0, 15, 30):
#     img2 = img1
#     img2 = myrotate(img2, deg)
#     img2 = myreshape(img2)
#
#     for lum in (-20, -10, 0, 10, 20):
#       img3= img2
#       img3 = mylum(img3, 1+lum/100.0)
#       # N = str(i+1) + '_' + str(deg) + '_' + str(lum)+'.jpg';
#       # if Label1=='L':
#       #   matplotlib.image.imsave(PathL+N, img3, cmap='gray',vmin=0,vmax=1)
#       # elif Label1=='R':
#       #   matplotlib.image.imsave(PathR + N, img3, cmap='gray', vmin=0, vmax=1)
#       Dataset[Count, 0] = Label1 == 'L'
#       Dataset[Count, 1:] = img3.flatten()
#       Count= Count+1
#
# numpy.savetxt("E:\PyProjects\data\Classification\Dataset.csv", Dataset, delimiter=',')





# img = mpimg.imread(r'E:\PyProjects\data\Classification\single_002.jpg')
# img = img[:, :, 0] / 255
#
# imgplot = plt.imshow(img,cmap='gray')
# plt.show()
# img= myrotate(img,30)
# imgplot = plt.imshow(img,cmap='gray')
# plt.show()
# img= myreshape(img)
# imgplot = plt.imshow(img,cmap='gray')
# plt.show()
# img = mylum(img, 1 + 0/100)
#
# imgplot = plt.imshow(img,cmap='gray')
# plt.show()
# print(img[0:5,0:5])





