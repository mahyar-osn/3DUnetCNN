import numpy as np
import cv2
import os
from PIL import Image
import tifffile
import nibabel as nib

filename = '53_2nd_Ori_SBack_C3.masked.nii'
#filename = 'Membrane_cell_label_masked.nii'


def read_images(dir_path):
    Images = []
    image_names = sorted(os.listdir(dir_path))
    for im in image_names:
        # image = cv2.imread(os.path.join(dir_path, im))
        # image_tmp = Image.open(os.path.join(dir_path, im))
        # image_tmp = tifffile.imread(os.path.join(dir_path, im))
        if im.endswith(filename):
            print(im)
            image = nib.load(os.path.join(dir_path, im))
            # image = np.asarray(image_tmp)
            # image = image_tmp.T
            Images.append(image)
    return Images


def image_size(image):
    return image.shape


def save_images(transformed, save_dir):
    if not (os.path.exists(save_dir)):
        os.mkdir(save_dir)
    k = 1
    for key, val in transformed.items():
        for i, j in enumerate(val):
            dir_name = str(k)
            path = os.path.join(save_dir, dir_name)
            if not os.path.exists(path):
                os.mkdir(path)
            nib.save(j, os.path.join(path, '53_2nd_Ori_SBack_C3' + '.nii.gz'))
            k+=1
            # tifffile.imsave(os.path.join(path, str(i + 1) + '.tiff'), j)


def Offset_op(input_length, output_length, stride):
    """
    Takes input(height, width), output(height, width) and strides

    :param input_length:
    :param output_length:
    :param stride:
    :return: offset, i.e. left out portion after applying strides
    """
    offset = (input_length) - (stride * ((input_length - output_length) // stride) + output_length)
    return offset


def Padding_op(Image, strides, offset_x, offset_y):
    """
    Takes an image, offset required to fit output image dimensions with given strides and calculates the
    padding it needs for perfect fit.

    :param Image:
    :param strides:
    :param offset_x:
    :param offset_y:
    :return: Padded image
    """
    padding_x = strides[0] - offset_x
    padding_y = strides[1] - offset_y
    # print(padding_x,padding_y)
    Padded_Image = np.zeros(shape=(Image.shape[0] + padding_x, Image.shape[1] + padding_y, Image.shape[2]),
                            dtype=Image.dtype)
    Padded_Image[padding_x // 2:(padding_x // 2) + (Image.shape[0]), padding_y // 2:(padding_y // 2) + Image.shape[1],
    :] = Image
    return Padded_Image


def Convolution_op(Image, size, strides):
    """
    Takes an image, Dimensions of the desired image and Strides.

    :param Image:
    :param size:
    :param strides:
    :return: List of cropped images
    """
    start_x = 0
    start_y = 0
    start_z = 0
    end_x = Image.shape[0] - size[0]
    end_y = Image.shape[1] - size[1]
    end_z = Image.shape[2] - size[2]

    n_rows = Image.shape[0] // strides[0]
    n_columns = Image.shape[1] // strides[1]
    n_depths = Image.shape[2] // strides[2]

    image_array = Image.get_fdata()
    small_images = []
    for i in range(n_rows):
        for j in range(n_columns):
            for k in range(n_depths):
                new_start_x = start_x + i * strides[0]
                new_start_y = start_y + j * strides[1]
                new_start_z = start_z + k * strides[2]
                # small_images.append(Image[new_start_x:new_start_x + size[0], new_start_y:new_start_y + size[1], :])
                small_image_temp = image_array[new_start_x:new_start_x + size[0], new_start_y:new_start_y + size[1], new_start_z:new_start_z + size[2]]
                small_image_temp_1 = np.transpose(small_image_temp, axes=[1,0,2])
                small_image_temp_2 = nib.Nifti1Image(small_image_temp_1, Image.affine)
                small_images.append(small_image_temp_2)
    return small_images


def transform(source_dir, size, strides=[None, None, None], PADDING=False):
    """
    Transforms the images/image into desired small images provided the strides
    If no strides are provided, the strides will default to the size of the desired image, i.e no overlapping will
    take place.

    :param source_dir:
    :param size:
    :param strides:
    :param PADDING:
    :return: dictionary with string of count starting from 1 as key and list of images as values.
    """
    if not (os.path.exists(source_dir)):
        raise Exception("Path does not exist!")
    else:
        im_path = None
        dir_path = None
        splits = source_dir.split('/')
        last = splits[-1].split('.')
        if len(last) > 1:
            im_path = source_dir
        else:
            dir_path = source_dir

        if im_path:
            Image = cv2.imread(im_path)
            Images = [Image]
        else:
            Images = read_images(source_dir)

        im_size = image_size(Images[0])
        num_images = len(Images)
        transformed_images = dict()
        # Images = np.array(Images)
        if PADDING:

            padded_images = []

            if strides[0] is None and strides[1] is None:
                strides[0] = size[0]
                strides[1] = size[1]
                offset_x = Images.shape[1] % size[0]
                offset_y = Images.shape[2] % size[1]
                for Image in Images:
                    Image_Padded = Padding_op(Image, strides, offset_x, offset_y)
                    padded_images.append(Image_Padded)

            elif strides[0] is None and strides[1] is not None:
                strides[0] = size[0]
                offset_x = Images.shape[1] % size[0]
                if strides[1] <= Images.shape[2]:
                    offset_y = Offset_op(Images.shape[2], size[1], strides[1])
                else:
                    print("stride_y must be between {0} and {1}".format(1, Images.shape[2] - size[1]))

                for Image in Images:
                    Image_Padded = Padding_op(Image, strides, offset_x, offset_y)
                    padded_images.append(Image_Padded)

            elif strides[0] is not None and strides[1] is None:

                strides[1] = size[1]
                offset_y = Images.shape[2] % size[1]

                if strides[0] <= Images.shape[1]:
                    offset_x = Offset_op(Images.shape[1], size[0], strides[0])
                else:
                    print("stride_x must be between {0} and {1}".format(1, Images.shape[1] - size[0]))

                for Image in Images:
                    Image_Padded = Padding_op(Image, strides, offset_x, offset_y)
                    padded_images.append(Image_Padded)
            else:
                if strides[0] > Images.shape[1] or strides[1] > Images.shape[2]:
                    print("stride_x must be between {0} and {1} and stride_y must be between {2} and {3}"
                          .format(1, Images.shape[1] - size[0], 1, Images.shape[2] - size[1]))

                else:
                    offset_x = Offset_op(Images.shape[1], size[0], strides[0])
                    offset_y = Offset_op(Images.shape[2], size[1], strides[1])

                for Image in Images:
                    Image_Padded = Padding_op(Image, strides, offset_x, offset_y)
                    padded_images.append(Image_Padded)

            count = 0
            for Image in padded_images:
                count += 1
                transformed_images[str(count)] = Convolution_op(Image, size, strides)

        else:
            if strides[0] is None and strides[1] is None:
                strides[0] = size[0]
                strides[1] = size[1]

            elif strides[0] is None and strides[1] is not None:
                strides[0] = size[0]

            elif strides[0] is not None and strides[1] is None:
                strides[1] = size[1]

            count = 0
            for Image in Images:
                count += 1
                transformed_images[str(count)] = Convolution_op(Image, size, strides)

        return transformed_images


def main():
    source_dir = '/hpc/nebr002/3DUnetCNN/Naz_data/53_2nd/single_image_striding/big'
    size = (128, 128, 128)
    strides = [88, 88, 88]
    padding = False
    grid_images = transform(source_dir, size, strides=strides, PADDING=padding)

    save_dir = '/hpc/nebr002/3DUnetCNN/Naz_data/53_2nd/single_image_striding/big/grids'
    save_images(grid_images, save_dir)


if __name__ == '__main__':
    main()
    print('done')
