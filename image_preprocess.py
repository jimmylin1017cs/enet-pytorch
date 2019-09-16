import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm

# define cityscape id to train id
id_to_train_id_map = {
    0:19,
    1:19,
    2:19,
    3:19,
    4:19,
    5:19,
    6:19,
    7:0,
    8:1,
    9:19,
    10:19,
    11:2,
    12:3,
    13:4,
    14:19,
    15:19,
    16:19,
    17:5,
    18:19,
    19:6,
    20:7,
    21:8,
    22:9,
    23:10,
    24:11,
    25:12,
    26:13,
    27:14,
    28:15,
    29:19,
    30:19,
    31:16,
    32:17,
    33:18,
    -1:-1,
}

# create the function to map train id
id_to_train_id_map_func = np.vectorize(id_to_train_id_map.get)

def create_output_image_path(output_image_path):

    if not os.path.exists(output_image_path):
        os.makedirs(output_image_path)

    if not os.path.exists(output_image_path + 'train'):
        os.makedirs(output_image_path + 'train')
    if not os.path.exists(output_image_path + 'trainannot'):
        os.makedirs(output_image_path + 'trainannot')

    if not os.path.exists(output_image_path + 'val'):
        os.makedirs(output_image_path + 'val')
    if not os.path.exists(output_image_path + 'valannot'):
        os.makedirs(output_image_path + 'valannot')

    if not os.path.exists(output_image_path + 'test'):
        os.makedirs(output_image_path + 'test')
    if not os.path.exists(output_image_path + 'testannot'):
        os.makedirs(output_image_path + 'testannot')
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-iptr', '--input-path-train',
                        type=str,
                        default='./cityscapes/train/',
                        help='The path to the train input dataset')

    parser.add_argument('-lptr', '--label-path-train',
                        type=str,
                        default='./cityscapes/trainannot/',
                        help='The path to the train label dataset')

    parser.add_argument('-ipv', '--input-path-val',
                        type=str,
                        default='./cityscapes/val/',
                        help='The path to the val input dataset')

    parser.add_argument('-lpv', '--label-path-val',
                        type=str,
                        default='./cityscapes/valannot/',
                        help='The path to the val label dataset')

    parser.add_argument('-iptt', '--input-path-test',
                        type=str,
                        default='./cityscapes/test/',
                        help='The path to the test input dataset')

    parser.add_argument('-lptt', '--label-path-test',
                        type=str,
                        default='./cityscapes/testannot/',
                        help='The path to the test label dataset')

    parser.add_argument('-oimh', '--output-image-height',
                        type=int,
                        default=256,
                        help='The output image height')

    parser.add_argument('-oimw', '--output-image-width',
                        type=int,
                        default=512,
                        help='The output image width')

    parser.add_argument('-op', '--output-image-path',
                        type=str,
                        default='./cityscapes_preprocess/',
                        help='The path to the output dataset')

    args = parser.parse_args()

    input_path_train = args.input_path_train
    label_path_train = args.label_path_train
    input_path_val = args.input_path_val
    label_path_val = args.label_path_val
    input_path_test = args.input_path_test
    label_path_test = args.label_path_test

    output_image_height = args.output_image_height # (the height all images fed to the model will be resized to)
    output_image_width = args.output_image_width # (the width all images fed to the model will be resized to)
    output_image_path = args.output_image_path

    number_of_classes = 20 # (number of object classes (road, sidewalk, car etc.))

    # create all output image directories
    create_output_image_path(output_image_path)

    '''
    print(input_path_train)
    print(label_path_train)
    print(input_path_val)
    print(label_path_val)
    print(input_path_test)
    print(label_path_test)
    print(output_image_height)
    print(output_image_width)
    print(output_image_path)
    '''

    if os.path.exists(input_path_train) and os.path.exists(label_path_train):

        input_train_names = os.listdir(input_path_train)
        input_train_names.sort()
        label_train_names = os.listdir(label_path_train)
        label_train_names.sort()

        for i in tqdm(range(len(input_train_names))):

            input_train_image_path = input_path_train + input_train_names[i]
            input_train_image = cv2.imread(input_train_image_path)
            input_train_image = cv2.resize(input_train_image, (output_image_width, output_image_height), interpolation=cv2.INTER_NEAREST)

            label_train_image_path = label_path_train + label_train_names[i]
            label_train_image = cv2.imread(label_train_image_path, cv2.IMREAD_GRAYSCALE)
            label_train_image = cv2.resize(label_train_image, (output_image_width, output_image_height), interpolation=cv2.INTER_NEAREST)
            label_train_image = id_to_train_id_map_func(label_train_image)

            cv2.imwrite(output_image_path + 'train/' + input_train_names[i], input_train_image)
            cv2.imwrite(output_image_path + 'trainannot/' +  label_train_names[i], label_train_image)
    else:
        print("The path to the train dataset not exist")

    if os.path.exists(input_path_val) and os.path.exists(label_path_val):

        input_val_names = os.listdir(input_path_val)
        input_val_names.sort()
        label_val_names = os.listdir(label_path_val)
        label_val_names.sort()

        for i in tqdm(range(len(input_val_names))):

            input_val_image_path = input_path_val + input_val_names[i]
            input_val_image = cv2.imread(input_val_image_path)
            input_val_image = cv2.resize(input_val_image, (output_image_width, output_image_height), interpolation=cv2.INTER_NEAREST)

            label_val_image_path = label_path_val + label_val_names[i]
            label_val_image = cv2.imread(label_val_image_path, cv2.IMREAD_GRAYSCALE)
            label_val_image = cv2.resize(label_val_image, (output_image_width, output_image_height), interpolation=cv2.INTER_NEAREST)
            label_val_image = id_to_train_id_map_func(label_val_image)

            cv2.imwrite(output_image_path + 'val/' + input_val_names[i], input_val_image)
            cv2.imwrite(output_image_path + 'valannot/' +  label_val_names[i], label_val_image)
    else:
        print("The path to the val dataset not exist")

    if os.path.exists(input_path_test) and os.path.exists(label_path_test):
        pass
    else:
        print("The path to the test dataset not exist")
    

    