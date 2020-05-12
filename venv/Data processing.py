import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2
from Net import net
import torch


img=mpimg.imread('E:\PyProjects\data\Test.jpg')
img = img/255

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


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


grayscale = rgb2gray(img)
grayscale= grayscale[120:180,120:190]
grayscale= myreshape(grayscale)

crop = grayscale
imgplot = plt.imshow(crop,cmap='gray')
plt.show()

net.load_state_dict(torch.load('E:\PyProjects\data\Classification\mymodel.pth'))
net.eval()

grayscale = torch.from_numpy(grayscale).view(1,1,90,-1).to(dtype=torch.float)
print(grayscale.size())
outputs = net(grayscale)

print(outputs)
# matplotlib.image.imsave('E:\PyProjects\data\Classification\single_058.jpg', crop,cmap='Greys')


