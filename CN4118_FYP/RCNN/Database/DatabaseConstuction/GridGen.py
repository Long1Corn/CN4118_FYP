import numpy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import json


def mygrid(N=20, single_size=100, deg=0, empty=0.0):
    img_size = 500
    # N 20*20 molecules images
    # single_size "diameter" of input images of single molecule
    # deg rotate deg
    # empty rate
    r = single_size // 2
    deg = deg * numpy.pi / 180

    with open(r'E:\PyProjects\CN4118_FYP\Database\dataset.txt', 'r') as in_file:
        dataset = json.load(in_file)

    img_w = numpy.ceil(single_size * (N + 0.5 + 0.5) * (numpy.cos(deg) + numpy.sin(deg))).astype(
        "int")  # additional 0.5 margin
    img_h = numpy.ceil(single_size * (3 ** 0.5 / 2 * (N - 1) + 1 + 0.5) * (numpy.cos(deg) + numpy.sin(deg))).astype(
        "int")  # additional 0.5 margin
    center = numpy.array([img_h // 2, img_w // 2])
    img = numpy.zeros([img_h, img_w])

    grid = numpy.zeros([N, N, 2])
    x0 = center[1] - single_size * (N / 2 - 0.5 + 0.25)
    y0 = center[0] + single_size * (N / 2 - 0.5) * 3 ** 0.5 / 2
    grid[0, 0] = numpy.array([y0, x0])

    for i in range(N):
        for j in range(N):
            x = x0 + j * single_size
            y = y0 - i * single_size * 3 ** 0.5 / 2
            if i % 2 == 1:
                x = x + 0.5 * single_size

            grid[i, j] = numpy.array([y, x])

    rotate = numpy.array([[numpy.cos(deg), -numpy.sin(deg)], [numpy.sin(deg), numpy.cos(deg)]])
    grid = center + numpy.dot((grid - center), rotate)
    grid = numpy.around(grid).astype("int")

    grid = grid.reshape([-1, 2])
    labels = [None] * N ** 2
    boxes = [None] * N ** 2
    c = 0

    for p in range(N ** 2):
        x = grid[p, 1]
        y = grid[p, 0]
        data, label = one_molecule(dataset, single_size, empty=empty)
        img[(y - r):(y + r), (x - r):(x + r)] += data

        if label != 0:
            labels[c] = label
            boxes[c] = numpy.array([x - r, y - r, x + r, y + r])
            c = c + 1

    labels = labels[:c]
    boxes = boxes[:c]
    return img, labels, boxes


def one_molecule(dataset, single_size, empty=0):
    p0 = numpy.random.random()
    if p0 < empty:
        return numpy.zeros([single_size, single_size]), 0

    L = len(dataset)
    p1 = numpy.random.randint(0, L)

    one = numpy.array(dataset[p1][1]).astype("uint8")
    label = dataset[p1][0]
    one = myreshape(one, desired_size=single_size)

    p3 = numpy.random.randint(0, 12)
    one = rotate_image(one, p3 * 30)
    return one, label


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


def rotate_image(image, angle):
    image_center = tuple(numpy.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


if __name__ == "__main__":

    molecules = 20
    single_size = 100
    deg = 0
    empty = 0

    img, labels, boxes = mygrid(N=molecules, single_size=single_size, deg=deg, empty=empty)
    plt.imshow(img, cmap="gray")
    ax = plt.gca()

    for i, box in enumerate(boxes):
        if labels[i] == 1:
            color = 'r'
        elif labels[i] == 2:
            color = 'y'

        x = (box[0] + box[2]) // 2
        y = (box[1] + box[3]) // 2
        r = (box[2] - box[0]) // 2
        circ = matplotlib.patches.Circle((x, y), r, linewidth=1, fill=False, color=color)
        ax.add_patch(circ)

    plt.show()

    plt.imshow(img, cmap="gray")
    plt.show()
