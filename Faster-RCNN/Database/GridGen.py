import numpy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import json
from detectron2.structures import BoxMode
import random


def mygrid(dataset, N=10, single_size=80, deg=0, empty=0.0):
    # N 20*20 molecules images
    # single_size "diameter" of input images of single molecule
    # deg rotate deg
    # empty rate
    r = single_size // 2
    deg = deg * numpy.pi / 180

    img_w = numpy.array(single_size * (N + 0.5 + 0.5))  # additional 0.5 margin
    img_h = numpy.array(single_size * (3 ** 0.5 / 2 * (N - 1) + 1 + 0.5))  # additional 0.5 margin
    img_w = numpy.ceil(img_w * abs(numpy.cos(deg)) + img_h * abs(numpy.sin(deg))).astype("int")
    img_h = numpy.ceil(img_h * abs(numpy.cos(deg)) + img_w * abs(numpy.sin(deg))).astype("int")

    center = numpy.array([img_h // 2, img_w // 2])
    img = numpy.zeros([img_h, img_w, 3])

    grid = numpy.zeros([N, N, 2])
    x0 = center[1] - single_size * (N / 2 - 0.5 + 0.25)
    y0 = center[0] - single_size * (N / 2 - 0.5) * 3 ** 0.5 / 2
    grid[0, 0] = numpy.array([y0, x0])

    for i in range(N):
        for j in range(N):
            x = x0 + j * single_size
            y = y0 + i * single_size * 3 ** 0.5 / 2
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

        if label != 0:
            r2 = r + 2
            img[(y - r2):(y + r2), (x - r2):(x + r2)] += data
            labels[c] = label
            boxes[c] = numpy.array([x - r2, y - r2, x + r2, y + r2])
            c = c + 1

    labels = labels[:c]
    boxes = boxes[:c]
    # padding = (output_size - img_h * output_size / img_w) // 2
    # boxes = [box * output_size / img_w + numpy.array([0, padding, 0, padding]) for box in boxes]
    img = noisy(img)

    return img, labels, boxes


def one_molecule(dataset, single_size, empty=0):
    p0 = numpy.random.rand(1)
    if p0 < empty:
        return numpy.zeros([single_size, single_size]), 0

    L = len(dataset)
    p1 = numpy.random.randint(0, L)

    label = dataset[p1][0]
    path = dataset[p1][1]

    one = cv2.imread(path)
    # one = numpy.array(dataset[p1][1]).astype("uint8")
    one = myreshape(one, desired_size=single_size + 4)

    p3 = numpy.random.randint(-45, 45)
    one = rotate_image(one, p3)

    p4 = (numpy.random.rand(1) - 0.5) * 0.8
    one = mylum(one, p4 + 1)
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


def noisy(image):
    row, col, ch = image.shape
    mean = 0
    var = 40
    sigma = var ** 0.5
    gauss = numpy.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return noisy


def mylum(img, lum):
    img = img * lum
    img = numpy.clip(img, 0, 255)
    return img


def Datagen(dataset, batch_size=4, molecules=5, rotate=0.0, empty=0.0, single_size=100):
    for b in range(batch_size):
        deg = 0
        if rotate:
            deg = numpy.random.randint(-rotate, rotate)
        empty = numpy.random.rand(1) * empty

        img, label, box = mygrid(dataset, N=molecules, single_size=single_size, deg=deg, empty=empty)

        L = len(label)
        boxes = [None] * L
        for num in range(L):
            boxes[num] = box[num].tolist()

    return img, label, boxes


def datadict_gen(dataset, path, size):
    dataset_dicts = []

    for i in range(size):
        molecules = numpy.random.randint(2, 10) * 2
        single_size = 100
        deg = 30
        empty = 0.03

        img, labels, boxes = Datagen(dataset, batch_size=1, molecules=molecules, single_size=single_size, rotate=deg,
                                     empty=empty)
        # img = img / 255
        img = img.clip(0, 255)

        path1 = path
        path2 = r"\\" + str(i).rjust(4, '0') + ".jpg"
        filename = path1 + path2
        # matplotlib.image.imsave(filename, img, cmap='gray')
        cv2.imwrite(filename, img)
        height, width = cv2.imread(filename).shape[:2]

        record = {}
        record['file_name'] = filename
        record['image_id'] = i
        record['height'] = height
        record['width'] = width

        objects = []
        for j in range(len(labels)):
            obj = {
                "bbox": boxes[j],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": labels[j] - 1,
                "iscrowd": 0
            }
            objects.append(obj)
        record["annotations"] = objects

        dataset_dicts.append(record)

    with open(path1 + r"\data_test.txt", 'w') as out_file:
        json.dump(dataset_dicts, out_file)


if __name__ == "__main__":
    data_qty = 32
    database_qty = 46
    idx = list(range(database_qty))
    random.shuffle(idx)

    with open(r'E:\PyProjects\Faster-RCNN\Single\dataset.txt', 'r') as in_file:
        dataset = json.load(in_file)
    dataset_train = [dataset[n] for n in idx[:data_qty]]
    dataset_test = [dataset[n] for n in idx[data_qty:]]

    path1 = r"E:\PyProjects\Faster-RCNN\Database\data32"

    datadict_gen(dataset_train, path1 + r"\train", 500)
    datadict_gen(dataset_test, path1 + r"\val", 100)

    # img (batch, 3, output_size, output_size)
    # labels list[batch]
    # boxes list[batch] 4

    # plt.imshow(img[0],cmap='gray')
    # ax = plt.gca()
    #
    # for i, box in enumerate(boxes[0]):
    #     if labels[0][i] == 1:
    #         color = 'r'
    #     elif labels[0][i] == 2:
    #         color = 'y'
    #
    #     x = (box[0] + box[2]) / 2 *300
    #     y = (box[1] + box[3]) / 2 *300
    #     r = (box[2] - box[0]) / 2 *300
    #     circ = matplotlib.patches.Circle((x, y), r, linewidth=1, fill=False, color=color)
    #     ax.add_patch(circ)
    #
    # plt.show()
    #
    # plt.imshow(img[0], cmap="gray")
    # plt.show()
