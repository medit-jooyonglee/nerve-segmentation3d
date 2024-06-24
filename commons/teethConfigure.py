import numpy as np
# from toothNet.config import Config
"""
[ref] Analysis of dimensions and shapes of maxillary and mandibular dental arch in Korean young adults

"""
# dental arch- max-length 75.0mm
# DENTAL_ARCH_M2W_MM = 75.0
DENTAL_ARCH_M2W_MM = 72.0


def compute_spacing(target_size):
    """
    [ref] https://jap.or.kr/Synapse/Data/PDFData/0170JAP/jap-9-321.pdf

    [refs] Analysis of dimensions and shapes of maxillary and mandibular dental arch in Korean young adults
    :param target_size:
    :return:
    """
    target_spacing = DENTAL_ARCH_M2W_MM / target_size
    return target_spacing

def get_itk2label():
    teeth_fdi2label_table = np.zeros([50], dtype=np.int)
    lab = 1
    for i in range(1, 5):
        for j in range(1, 9):
            tau = i * 10 + j

            teeth_fdi2label_table[tau] = lab
            lab += 1
    # TODO:writing unknow label?
    unknow_label = 1
    teeth_fdi2label_table[unknow_label] = 0
    return teeth_fdi2label_table



def get_label2fdi():
    """
    from int to fdi-number
    :return:
    """
    lab = 1
    total_teeth = 32
    labels = np.zeros([total_teeth+1], dtype=np.int32)
    for i in range(1, 5):
        for j in range(1, 9):
            tau = i * 10 + j
            labels[lab] = tau
            lab += 1
    return labels

def get_fdi2label():
    """
    from fdi-number to int
    :return:
    """
    lab = 1
    # total_teeth = 32 + 1
    labels = np.zeros([50], dtype=np.int32)
    for i in range(1, 5):
        for j in range(1, 9):
            tau = i * 10 + j
            labels[tau] = lab
            lab += 1
    return labels



def get_label_to_2classification(with_background=False):
    """
    incisor(1~2), canine(3), premolar(4,5) molar(6, 7, 8)
    and upper and lower
    total 8 classification + 1(bg)

    :param without_background:
    :type without_background:
    :return:
    :rtype:
    """

    increment = int(with_background)
    teeth_index = {
        0: 0,
        1: 0,
        2: 0,
        3: 1,
        4: 2,
        5: 2,
        6: 3,
        7: 3,
        8: 3
    }

    for k in teeth_index.keys():
        teeth_index[k] += increment

    # fdi2classlabel = {}
    fdi2label = get_fdi2label()
    lab2_2class = np.zeros([32+1], dtype=np.int32)
    for i in range(1, 5):
        for j in range(1, 9):
            tau = i * 10 + j
            pose = tau // 10
            category = tau % 10
            ipose = 0 if pose < 3 else 1
            jteeth = teeth_index[category]

            # add 1 because of background(=0)
            _2label = ipose +1
            label = fdi2label[tau]
            # print(tau, label, _4label)
            lab2_2class[label] = _2label
    return lab2_2class


def get_label_to_8classification(with_background=False):
    """
    incisor(1~2), canine(3), premolar(4,5) molar(6, 7, 8)
    and upper and lower
    total 8 classification + 1(bg)
    :return:
    """

    teeth_index = {
        0: 0,
        1: 0,
        2: 0,
        3: 1,
        4: 2,
        5: 2,
        6: 3,
        7: 3,
        8: 3
    }

    increment = int(with_background)
    for k in teeth_index.keys():
        teeth_index[k] += increment

    # fdi2classlabel = {}
    fdi2label = get_fdi2label()
    lab2_4class = np.zeros([32+1], dtype=np.int32)
    for i in range(1, 5):
        for j in range(1, 9):
            tau = i * 10 + j
            pose = tau // 10
            category = tau % 10
            ipose = 0 if pose < 3 else 1
            jteeth = teeth_index[category]

            # add 1 because of background(=0)
            _4label = 4*ipose + jteeth + 1
            label = fdi2label[tau]
            # print(tau, label, _4label)
            lab2_4class[label] = _4label
    return lab2_4class



def get_label_to_4classification(with_background=False):
    """
    incisor(1~2), canine(3), premolar(4,5) molar(6, 7, 8)
    and upper and lower
    total 8 classification + 1(bg)
    :return:
    """

    teeth_index = {
        0: 0,
        1: 0,
        2: 0,
        3: 1,
        4: 1,
        5: 1,
        6: 1,
        7: 1,
        8: 1
    }

    increment = int(with_background)
    for k in teeth_index.keys():
        teeth_index[k] += increment

    # fdi2classlabel = {}
    fdi2label = get_fdi2label()
    lab2_4class = np.zeros([32+1], dtype=np.int32)
    for i in range(1, 5):
        for j in range(1, 9):
            tau = i * 10 + j
            pose = tau // 10
            category = tau % 10
            ipose = 0 if pose < 3 else 1
            jteeth = teeth_index[category]

            # add 1 because of background(=0)
            _4label = 2*ipose + jteeth + 1
            label = fdi2label[tau]
            # print(tau, label, _4label)
            lab2_4class[label] = _4label
    return lab2_4class


def get_teeth_color_table(normalize=True):
    "https://github.com/tensorflow/models/blob/master/research/deeplab/utils/get_dataset_colormap.py"
    color_table = np.asarray([
        [255, 255, 255],  # bg
        [6, 230, 230],    # 1
        [80, 50, 50],       #2
        [4, 200, 3],        #3
        [30, 20, 240],     #4
        [240, 10, 7],    #5
        [224, 5, 255],  #6
        [235, 255, 7],  #7
        [150, 5, 61],   #8
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        ])
    if normalize:
        color_table = color_table/255.

    return color_table

def _get_teeth_table_img():
    colors = get_teeth_color_table()
    width = 100
    height = 100
    tableimg = []
    for i in range(1, 9):
        img = np.ones([width, height, 3]) * colors[i] * 255
        tableimg.append(img)
        print(i, colors[i]*255)
    tableimg = np.concatenate(tableimg, axis=1)
    # tableimg = tableimg[:, ::-1, :]
    print(tableimg.shape)
    import cv2
    tableimg = tableimg.astype(np.uint8)
    cv2.imwrite("data/colortable.png", tableimg[:, :, ::-1])

def test():
    _get_teeth_table_img()

    # print(get_label_to_4classification())
if __name__=="__main__":
    test()
