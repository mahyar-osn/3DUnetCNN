import os, sys
import glob
import numpy as np

sys.path.append("../unet3d")
from train_isensee2017 import config
from normalize import normalize_data_storage, reslice_image_set
from prediction import run_validation_cases, predict_from_data_file_and_write_image,prediction_to_image, predict, patch_wise_prediction
from training import load_old_model
from utils import pickle_load
import tables
import h5py
import pickle
import nibabel as nib
import tifffile as tf


def create_predict_file():
    data_files = list()
    subject_ids = list()
    for subject_dir in glob.glob(os.path.join(os.path.dirname(__file__), "data", "preprocessed", "Predict", "*")):
        subject_ids.append(os.path.basename(subject_dir))
        subject_files = []
        for modality in config["predict_modalities"]:
            imfile = os.path.join(subject_dir, modality + ".nii.gz")
            im = nib.load(imfile)
            imarray = im.get_fdata()
            mean_im = np.mean(imarray)
            if mean_im > 0:
                subject_files.append(os.path.join(subject_dir, modality + ".nii.gz"))
            else:
                if os.path.basename(subject_dir) in subject_ids:
                    subject_ids.remove(os.path.basename(subject_dir))
        if len(subject_files) < 1:
            pass
        else:
            data_files.append(tuple(subject_files))

    write_data_to_file(data_files,
                       config["predict_file"],
                       image_shape=config["image_shape"],
                       subject_ids=subject_ids)
    return subject_ids


def create_data_file(out_file, n_channels, n_samples, image_shape):
    hdf5_file = tables.open_file(out_file, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    data_shape = tuple([0, n_channels] + list(image_shape))
    data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape,
                                           filters=filters, expectedrows=n_samples)
    affine_storage = hdf5_file.create_earray(hdf5_file.root, 'affine', tables.Float32Atom(), shape=(0, 4, 4),
                                             filters=filters, expectedrows=n_samples)
    return hdf5_file, data_storage, affine_storage


def write_image_data_to_file(image_files, data_storage, image_shape, n_channels, affine_storage, crop=True):

    for set_of_files in image_files:
        images = reslice_image_set(set_of_files, image_shape, label_indices=len(set_of_files) - 1, crop=crop)
        subject_data = [image.get_data() for image in images]
        add_data_to_storage(data_storage, affine_storage, subject_data, images[0].affine, n_channels)

    return data_storage


def add_data_to_storage(data_storage, affine_storage, subject_data, affine, n_channels):
    data_storage.append(np.asarray(subject_data[:n_channels])[np.newaxis])
    affine_storage.append(np.asarray(affine)[np.newaxis])


def write_data_to_file(training_data_files, out_file, image_shape, subject_ids=None,
                       normalize=True, crop=True):

    n_samples = len(training_data_files)
    n_channels = len(training_data_files[0])

    try:
        hdf5_file, data_storage, affine_storage = create_data_file(out_file,
                                                                   n_channels=n_channels,
                                                                   n_samples=n_samples,
                                                                   image_shape=image_shape)
    except Exception as e:
        os.remove(out_file)
        raise e
    write_image_data_to_file(training_data_files, data_storage, image_shape,
                             n_channels=n_channels, affine_storage=affine_storage, crop=crop)
    if subject_ids:
        hdf5_file.create_array(hdf5_file.root, 'subject_ids', obj=subject_ids)
    if normalize:
        normalize_data_storage(data_storage)
    hdf5_file.close()
    return out_file


def main():
    if not os.path.exists(config["predict_file"]):
        subject_ids = create_predict_file()
        with open("predict_ids.pkl", "wb") as idfile:
            pickle.dump(subject_ids, idfile)

    model = load_old_model(config["model_file"])
    # model = None
    test_data_read_h5 = tables.open_file(config["predict_file"], "r")
    test_data_tmp = np.asarray([test_data_read_h5.root.data])
    test_data = test_data_tmp[0]

    training_modalities = config["training_modalities"]
    prediction_dir = os.path.abspath("prediction")
    output_dir = prediction_dir

    affine = test_data_read_h5.root.affine[0]
    permute = False
    labels = config["labels"]
    output_label_map = True
    threshold = 0.5
    overlap = 16

    with open("predict_ids.pkl", "rb") as idfile:
        list_cases = pickle.load(idfile)

    num_samples = 0
    for case in list_cases:
        case_directory = os.path.join(output_dir, str(case))

        print("Predicting image {0}".format(case))

        if not os.path.exists(case_directory):
            os.makedirs(case_directory)

        test_data_reshaped = test_data[num_samples].reshape(1, test_data[0].shape[0], test_data[0].shape[1],
                                                            test_data[0].shape[2], test_data[0].shape[3])

        for i, modality in enumerate(training_modalities):
            tf.imsave(os.path.join(case_directory, "data_{0}.tiff".format(modality)),
                      np.transpose(test_data_reshaped[0, i], axes=[2, 0, 1]))

            image = nib.Nifti1Image(test_data_reshaped[0, i], affine)
            image.to_filename(os.path.join(case_directory, "data_{0}.nii".format(modality)))

        patch_shape = tuple([int(dim) for dim in model.input.shape[-3:]])

        if patch_shape == test_data.shape[-3:]:
            prediction = predict(model, test_data_reshaped, permute=permute, batch=config["validation_batch_size"])
        else:
            prediction = patch_wise_prediction(model=model, data=test_data_reshaped, overlap=overlap, permute=permute)[np.newaxis]

        prediction_image = prediction_to_image(prediction, affine, label_map=output_label_map, threshold=threshold,
                                               labels=labels)

        prediction_image_tiff = prediction_image.get_fdata()

        if isinstance(prediction_image, list):
            for i, image in enumerate(prediction_image):
                image.to_filename(os.path.join(case_directory, "prediction_{0}.nii".format(i + 1)))
        else:
            tf.imsave(os.path.join(case_directory, "prediction.tiff"), np.transpose(prediction_image_tiff, axes=[2,0,1]))
            prediction_image.to_filename(os.path.join(case_directory, "prediction.nii"))

        num_samples+=1

    return None


if __name__ == "__main__":
    main()