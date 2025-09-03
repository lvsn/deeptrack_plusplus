import sys
import os
from multiprocessing import cpu_count

from pytorch_toolbox.train_loop import TrainLoop
import torch
from torch.utils import data

from pytorch_toolbox.io import yaml_load, yaml_dump
import numpy as np
from pytorch_toolbox.transformations.compose import Compose

from deep_6dof_tracking.data.data_augmentation import Occluder, HSVNoise, Background, GaussianNoise, \
    GaussianBlur, OffsetDepth, NormalizeChannels, ToTensor
from deep_6dof_tracking.data.deeptrack_bin_loader import DeepTrackBinLoader
from deep_6dof_tracking.data.deeptrack_bin_mask_loader import DeepTrackBinMaskLoader
from deep_6dof_tracking.networks.deeptrack_bin_channeldrop_net import DeepTrackBinChannelDropNet
from deep_6dof_tracking.networks.deeptrack_bin_net import DeepTrackBinNet
from deep_6dof_tracking.test.callback_validation import DeepTrackBinCallbackValidation


class DummyAugment(object):
    def __call__(self, data):
        rgbA, depthA, rgbB, depthB, prior = data
        return rgbA, depthA, rgbB, depthB, prior, None

if __name__ == '__main__':

    #
    #   Load configurations
    #
    try:
        config_path = sys.argv[1]
    except IndexError:
        config_path = "validation_config.yml"
    configs = yaml_load(config_path)

    data_path = configs["data_path"]
    backend = configs["backend"]
    batch_size = configs["batch_size"]
    epochs = configs["epochs"]
    use_shared_memory = configs["use_shared_memory"]
    number_of_core = configs["number_of_core"]
    learning_rate = configs["learning_rate"]
    occluder_path = configs["occluder_path"]
    background_path = configs["background_path"]
    model_path = configs["model_path"]

    architecture = configs["architecture"]

    if number_of_core == -1:
        number_of_core = cpu_count()

    #
    #   Instantiate models/loaders/etc.
    #
    loader_param = {}

    if architecture == "bin":
        callbacks = DeepTrackBinCallbackValidation()
        n_bin = 41
        loader_param["n_bin"] = n_bin
        loader_param["callback"] = callbacks
        model = DeepTrackBinNet(n_bin)
        loader_class = DeepTrackBinLoader

    elif architecture == "bin_cauchy":
        callbacks = DeepTrackBinCallbackValidation()
        n_bin = 21
        loader_param["n_bin"] = n_bin
        loader_param["callback"] = callbacks
        loader_param["linear_bins"] = False
        model = DeepTrackBinNet(n_bin)
        loader_class = DeepTrackBinLoader

    elif architecture == "bin_mask":
        callbacks = DeepTrackBinCallbackValidation()
        n_bin = 41
        loader_param["n_bin"] = n_bin
        loader_param["callback"] = callbacks
        model = DeepTrackBinNet(n_bin)
        loader_class = DeepTrackBinMaskLoader

    elif architecture == "bin_channeldrop":
        callbacks = DeepTrackBinCallbackValidation()
        n_bin = 41
        loader_param["n_bin"] = n_bin
        loader_param["callback"] = callbacks
        model = DeepTrackBinChannelDropNet(n_bin)
        loader_class = DeepTrackBinLoader

    # Here we use the following transformations:
    images_mean = np.load(os.path.join(data_path, "mean.npy"))
    images_std = np.load(os.path.join(data_path, "std.npy"))
    # transfformations are a series of transform to pass to the input data. Here we have to build a list of
    # transforms for each inputs to the network's forward call

    pretransforms = [DummyAugment()]

    posttransforms = [Compose([Background(background_path),
                               OffsetDepth(),
                               NormalizeChannels(images_mean, images_std),
                               ToTensor()])]

    print("Load datasets from {}".format(data_path))
    valid_dataset = loader_class(os.path.join(data_path, "valid"), pretransforms, posttransforms, **loader_param)

    valid_dataset.data_pose = valid_dataset.data_pose[:500]
    valid_dataset.imgs = valid_dataset.data_pose

    val_loader = data.DataLoader(valid_dataset,
                                 batch_size=batch_size,
                                 num_workers=number_of_core,
                                 pin_memory=use_shared_memory,
                                 )

    # Instantiate the train loop and train the model.
    train_loop_handler = TrainLoop(model, None, val_loader, None, backend)
    train_loop_handler.add_callback(callbacks)

    for i in [33, 35, 40]:
        checkpoint_path = os.path.join(model_path, "checkpoint{}.pth.tar".format(i))
        print("Load model path : {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])

        print("Test Begins:")
        train_loop_handler.validate()

