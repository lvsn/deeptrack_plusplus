import configparser
import sys
import os
from multiprocessing import cpu_count
import argparse
import json

from deep_6dof_tracking.callbacks_da import DeepTrackCallbackDA
from deep_6dof_tracking.data.deeptrack_loader_moddropout import DeepTrackLoaderModdropout
from deep_6dof_tracking.data.deeptrack_loader_rgb import DeepTrackLoaderRGB
from deep_6dof_tracking.networks.deeptrack_res_net import DeepTrackResNet
from deep_6dof_tracking.networks.deeptrack_res_net_consistence import DeepTrackConsistence
from deep_6dof_tracking.networks.deeptrack_res_net_deep import DeepTrackResNetDeep
from deep_6dof_tracking.networks.deeptrack_res_net_grad import DeepTrackResNetGrad
from deep_6dof_tracking.networks.deeptrack_res_net_mask import DeepTrackResNetMask
from deep_6dof_tracking.networks.deeptrack_res_net_moddrop import DeepTrackResNetModDrop
from deep_6dof_tracking.networks.deeptrack_res_net_moddrop2 import DeepTrackResNetModDrop2
from deep_6dof_tracking.networks.deeptrack_res_net_moddrop3 import DeepTrackResNetModDrop3
from deep_6dof_tracking.networks.deeptrack_res_net_moddrop_deep import DeepTrackResNetModDropDeep
from deep_6dof_tracking.networks.deeptrack_res_net_moddrop_groupnorm import DeepTrackResNetModDropGroupNorm
from deep_6dof_tracking.networks.deeptrack_res_net_splitstream import DeepTrackResNetSplit
from deep_6dof_tracking.networks.old.deeptrack_dense_net import DeepTrackDenseNet
from deep_6dof_tracking.networks.FPNet import RefineNet

from pytorch_toolbox.train_loop import TrainLoop
import torch
from torch import optim
from torch.utils import data
from torch.optim import lr_scheduler

from pytorch_toolbox.io import yaml_load, yaml_dump
import numpy as np
from pytorch_toolbox.transformations.compose import Compose

from deep_6dof_tracking.callback_flow import DeepTrackFlowCallback
from deep_6dof_tracking.callback_geo_flow import DeepTrackGeoFlowCallback
from deep_6dof_tracking.callbacks import DeepTrackCallback
from deep_6dof_tracking.data.data_augmentation import Occluder, HSVNoise, Background, GaussianNoise, \
    GaussianBlur, OffsetDepth, NormalizeChannels, ToTensor, ChannelHide, DepthDownsample, KinectOffset, HideRGB, HideDepth, BoundingBox3D
from deep_6dof_tracking.data.deeptrack_composition_loader import DeepTrackCompositionLoader
from deep_6dof_tracking.data.deeptrack_flow_loader import DeepTrackFlowLoader
from deep_6dof_tracking.data.deeptrack_geo_flow_loader import DeepTrackGeoFlowLoader
from deep_6dof_tracking.data.deeptrack_geo_loader import DeepTrackGeoLoader
from deep_6dof_tracking.data.deeptrack_loader import DeepTrackLoader
from deep_6dof_tracking.data.deeptrack_loader_angle import DeepTrackAngleLoader
from deep_6dof_tracking.data.deeptrack_mask_loader import DeepTrackMaskLoader
from deep_6dof_tracking.data.deeptrack_z_loader import DeepTrackZLoader
from deep_6dof_tracking.networks.deeptrack_res_net_2stream import DeepTrackResNet2Stream
from deep_6dof_tracking.callback_geo import DeepTrackGeoCallback
from deep_6dof_tracking.utils.camera import Camera
from deep_6dof_tracking.networks.PMLoss import PMLoss
from deep_6dof_tracking.data.modelrenderer2 import ModelRenderer2

if __name__ == '__main__':
    #
    #   load configurations from
    #

    parser = argparse.ArgumentParser(description='Train DeepTrack')
    parser.add_argument('-o', '--output', help="Output path", metavar="FILE")
    parser.add_argument('-d', '--dataset', help="Dataset path", metavar="FILE")
    parser.add_argument('-b', '--background', help="Background path", metavar="FILE")
    parser.add_argument('-r', '--occluder', help="Occluder path", metavar="FILE")
    parser.add_argument('-f', '--finetune', help="finetune path", default="None")
    parser.add_argument('-c', '--from_last', help="Continue training from last checkpoint", action="store_true")
    parser.add_argument('-a', '--architecture', help="architecture name", action="store", default="squeeze_large") #res

    parser.add_argument('-i', '--device', help="Gpu id", action="store", default=0, type=int)
    parser.add_argument('-w', '--weightdecay', help="weight decay", action="store", default=0.000001, type=float)
    parser.add_argument('-l', '--learningrate', help="learning rate", action="store", default=0.001, type=float)
    parser.add_argument('-k', '--backend', help="backend : cuda | cpu", action="store", default="cuda")
    parser.add_argument('-e', '--epoch', help="number of epoch", action="store", default=25, type=int)
    parser.add_argument('-s', '--batchsize', help="Size of minibatch", action="store", default=128, type=int)
    parser.add_argument('-m', '--sharememory', help="Activate share memory", action="store_true")
    parser.add_argument('-n', '--ncore', help="number of cpu core to use, -1 is all core", action="store", default=-1, type=int)
    parser.add_argument('-g', '--gradientclip', help="Activate gradient clip", action="store_true")
    parser.add_argument('--tensorboard', help="Size of minibatch", action="store_true")
    parser.add_argument('--phase', help="Just a way to pass an argument to the network ", default=0, type=int)
    parser.add_argument('--depthonly', help="Use only depth for training", action="store_true")
    parser.add_argument('--rgbonly', help="Use only rgb for training", action="store_true")
    parser.add_argument('--depthnet', help="Size of minibatch", action="store_true")
    parser.add_argument('--delta_pose', help="Network predicts image coordinates translation", action="store_true")
    parser.add_argument('--pml', help="Use DeepIM's Point Matching Loss", action="store_true")
    parser.add_argument('--camera', help="path to camera file", action="store", default="./data/sensors/camera_parameter_files/synthetic.json")
    parser.add_argument('--shader', help="path to shader files", action="store", default="./data/shaders")
    parser.add_argument('--no_token', help="transformer token thing", action="store_true")
    parser.add_argument('--hybrid_vit', help="transformer token thing", action="store_true")
    parser.add_argument('--more_heads', help="transformer head thing", action="store_true")
    parser.add_argument('--smaller', help="transformer size thing", action="store_true")
    parser.add_argument('--newbg', help="use new and 'improved' background", action="store_true")
    parser.add_argument('--newbg2', help="use new and 'improved' background (2)", action="store_true")
    parser.add_argument('--newbg3', help="use new and 'improved' background (3)", action="store_true")
    parser.add_argument('--bb3d', help="crop the observed data with a 3D bounding box", action="store_true")
    parser.add_argument('--bb3d_rgb', help="the pixels cropped by the 3D BB will be put to 0 in the rgb image", action="store_true")
    parser.add_argument('--bb_padding', help="padding for the 3D bounding box", action="store", default=0.5, type=float)
    parser.add_argument('--same_mean', help="use the same mean and std for render and obs data", action="store_true")
    parser.add_argument('--hide_prob', help="probability of hiding a channel", action="store", default=0.3, type=float)
    parser.add_argument('--add_dilation', help="add dilation to replicate sensor noise", action="store_true")

    parser.add_argument('--config', help="config file", metavar="FILE")

    arguments = parser.parse_args()

    learning_rate = arguments.learningrate
    weight_decay = arguments.weightdecay
    device_id = arguments.device
    backend = arguments.backend
    epochs = arguments.epoch
    batch_size = arguments.batchsize
    use_shared_memory = arguments.sharememory
    number_of_core = arguments.ncore
    gradient_clip = arguments.gradientclip
    architecture = arguments.architecture
    start_from_last = arguments.from_last
    use_tensorboard = arguments.tensorboard
    phase = arguments.phase
    depthonly = arguments.depthonly
    rgbonly = arguments.rgbonly
    depthnet = arguments.depthnet

    output_path = arguments.output
    occluder_path = arguments.occluder
    background_path = arguments.background
    data_path = arguments.dataset
    finetune_path = arguments.finetune
    delta_pose = arguments.delta_pose
    pml = arguments.pml
    camera_path = arguments.camera
    shader_path = arguments.shader
    no_token = arguments.no_token
    hybrid_vit = arguments.hybrid_vit
    more_heads = arguments.more_heads
    smaller = arguments.smaller
    newbg = arguments.newbg
    newbg2 = arguments.newbg2
    newbg3 = arguments.newbg3
    bb3d = arguments.bb3d
    bb3d_rgb = arguments.bb3d_rgb
    bb_padding = arguments.bb_padding
    same_mean = arguments.same_mean
    hide_prob = arguments.hide_prob
    add_dilation = arguments.add_dilation

    if arguments.config is not None:
        config = configparser.ConfigParser()
        config.read(arguments.config)

        # DEFAULT
        occluder_path = config['DEFAULT']['occluder']
        background_path = config['DEFAULT']['background']
        data_path = config['DEFAULT']['dataset']
        output_path = config['DEFAULT']['output']

        # RESSOURCES
        use_shared_memory = config['RESSOURCES'].getboolean('sharememory')
        number_of_core = config['RESSOURCES'].getint('ncore')
        device_id = config['RESSOURCES'].getint('device')
        backend = config['RESSOURCES']['backend']

        # TRAINING
        learning_rate = config['TRAINING'].getfloat('learningrate')
        weight_decay = config['TRAINING'].getfloat('weightdecay')
        epochs = config['TRAINING'].getint('epoch')
        batch_size = config['TRAINING'].getint('batchsize')
        gradient_clip = config['TRAINING'].getboolean('gradientclip')
        architecture = config['TRAINING']['architecture']

        # OTHER
        start_from_last = config['OTHER'].getboolean('from_last')
        use_tensorboard = config['OTHER'].getboolean('tensorboard')
        phase = config['OTHER'].getint('phase')
        depthonly = config['OTHER'].getboolean('depthonly')
        rgbonly = config['OTHER'].getboolean('rgbonly')
        depthnet = config['OTHER'].getboolean('depthnet')
        finetune_path = config['OTHER']['finetune']
        camera_path = config['OTHER']['camera']
        shader_path = config['OTHER']['shader']

        # PML
        pml = config['PML'].getboolean('pml')
        delta_pose = config['PML'].getboolean('delta_pose')

        # VIT
        no_token = config['VIT'].getboolean('no_token')
        hybrid_vit = config['VIT'].getboolean('hybrid_vit')
        more_heads = config['VIT'].getboolean('more_heads')
        smaller = config['VIT'].getboolean('smaller')

        # DATA
        newbg = config['DATA'].getboolean('newbg')
        newbg2 = config['DATA'].getboolean('newbg2')
        newbg3 = config['DATA'].getboolean('newbg3')
        bb3d = config['DATA'].getboolean('bb3d')
        bb3d_rgb = config['DATA'].getboolean('bb3d_rgb')
        bb_padding = config['DATA'].getfloat('bb_padding')
        same_mean = config['DATA'].getboolean('same_mean')
        hide_prob = config['DATA'].getfloat('hide_prob')
        add_dilation = config['DATA'].getboolean('add_dilation')


    #
    #   Load configurations from file
    #
    data_path = os.path.expandvars(data_path)
    output_path = os.path.expandvars(output_path)
    occluder_path = os.path.expandvars(occluder_path)
    background_path = os.path.expandvars(background_path)
    finetune_path = os.path.expandvars(finetune_path)

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if number_of_core == -1:
        number_of_core = cpu_count()
    if backend == "cuda":
        torch.cuda.set_device(device_id)
    tensorboard_path = ""
    if use_tensorboard:
        tensorboard_path = os.path.join(output_path, "tensorboard_logs")

    #
    #   Instantiate models/loaders/etc.
    #
    loader_param = {}
    if architecture == "fpnet":
        model_class = RefineNet
        callbacks = DeepTrackCallback(output_path, is_dof_only=True)
        loader_class = DeepTrackLoader
    elif architecture == "res":
        model_class = DeepTrackResNet
        callbacks = DeepTrackCallback(output_path, is_dof_only=True)
        loader_class = DeepTrackLoader
    if architecture == "res_moddrop":
        model_class = DeepTrackResNetModDrop
        callbacks = DeepTrackCallback(output_path, is_dof_only=True)
        loader_class = DeepTrackLoader
    elif architecture == "res_moddropout_deep":
        model_class = DeepTrackResNetModDropDeep
        callbacks = DeepTrackCallback(output_path, is_dof_only=True)
        loader_class = DeepTrackLoader
    elif architecture == "res_deep":
        model_class = DeepTrackResNetDeep
        callbacks = DeepTrackCallback(output_path, is_dof_only=True)
        loader_class = DeepTrackLoader
    elif architecture == "res_moddropout_groupnorm":
        model_class = DeepTrackResNetModDropGroupNorm
        callbacks = DeepTrackCallback(output_path, is_dof_only=True)
        loader_class = DeepTrackLoader
    elif architecture == "res_moddrop2":
        model_class = DeepTrackResNetModDrop2
        callbacks = DeepTrackCallback(output_path, is_dof_only=True)
        loader_class = DeepTrackLoader
    elif architecture == "res_moddrop3":
        model_class = DeepTrackResNetModDrop3
        callbacks = DeepTrackCallback(output_path, is_dof_only=True)
        loader_class = DeepTrackLoader
    elif architecture == "res_2stream":
        model_class = DeepTrackResNet2Stream
        callbacks = DeepTrackCallback(output_path, is_dof_only=True)
        loader_class = DeepTrackLoader
    elif architecture == "res_consistence":
        model_class = DeepTrackConsistence
        callbacks = DeepTrackCallbackDA(output_path, is_dof_only=True)
        loader_class = DeepTrackLoader
    elif architecture == "res_split":
        model_class = DeepTrackResNetSplit
        callbacks = DeepTrackCallback(output_path, is_dof_only=True)
        if phase==1:
            callbacks = DeepTrackCallbackDA(output_path, is_dof_only=True)
        loader_class = DeepTrackLoader
    elif architecture == "res_mask":
        model_class = DeepTrackResNetMask
        callbacks = DeepTrackFlowCallback(output_path, is_dof_only=True)
        loader_class = DeepTrackMaskLoader

    # Here we use the following transformations:
    images_mean = np.load(os.path.join(data_path, "mean.npy"))
    images_std = np.load(os.path.join(data_path, "std.npy"))
    if same_mean:
        images_mean[:4] = images_mean[4:]
        images_std[:4] = images_std[4:]

    with open(os.path.join(data_path, "train", "viewpoints.json")) as data_file:
        metadata = json.load(data_file)["metaData"]
    
    # transfformations are a series of transform to pass to the input data. Here we have to build a list of
    # transforms for each inputs to the network's forward call

    print(images_mean)
    print(images_std)

    pretransforms = [Compose([Occluder(occluder_path, 0.75)])]

    posttransforms = [HSVNoise(0.07, 0.05, 0.1),
                               #KinectOffset(),
                               Background(background_path, newbg=newbg, newbg2=newbg2, newbg3=newbg3, add_dilation=add_dilation),
                               OffsetDepth(),
                               GaussianNoise(2, 5),
                               GaussianBlur(6),
                               DepthDownsample(0.7)]
    
    if not rgbonly and not depthonly:
        posttransforms.append(ChannelHide(disable_proba=hide_prob))

    if bb3d:
        posttransforms.append(BoundingBox3D(object_max_width=float(metadata['bounding_box_width']), original_padding=0.15, padding=bb_padding, crop_rgb=bb3d_rgb))

    posttransforms.append(NormalizeChannels(images_mean, images_std))

    if rgbonly:
        posttransforms.append(HideDepth())
    
    if depthonly:
        posttransforms.append(HideRGB())

    posttransforms.append(ToTensor())
    print(posttransforms)
    posttransforms = [Compose(posttransforms)]
    
    print("Load datasets from {}".format(data_path))
    train_dataset = loader_class(os.path.join(data_path, "train"), pretransforms, posttransforms, **loader_param)
    valid_dataset = loader_class(os.path.join(data_path, "valid"), pretransforms, posttransforms, **loader_param)

    # Save important information to output:
    print("Save meta data in {}".format(output_path))
    np.save(os.path.join(output_path, "mean.npy"), images_mean)
    np.save(os.path.join(output_path, "std.npy"), images_std)
    train_dataset.metadata["bb3d"] = bb3d
    yaml_dump(os.path.join(output_path, "meta.yml"), train_dataset.metadata)

    # Instantiate the data loader needed for the train loop. These use dataset object to build random minibatch
    # on multiple cpu core
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=number_of_core,
                                   pin_memory=use_shared_memory,
                                   drop_last=True,
                                   )

    val_loader = data.DataLoader(valid_dataset,
                                 batch_size=batch_size,
                                 num_workers=number_of_core,
                                 pin_memory=use_shared_memory,
                                 )

    # Setup model
    camera = Camera.load_from_json(camera_path)
    if pml:
        models = yaml_load(os.path.join(data_path, "train", "models.yml"))['models']
        geometry_path = os.path.join(models[0]["path"], "geometry.ply")
        window_size = [camera.width, camera.height]
        bounding_box_width = float(train_dataset.metadata['bounding_box_width'])
        model_renderer = ModelRenderer2(geometry_path, shader_path, camera, [window_size, (174, 174)], object_max_width=bounding_box_width)
        loss_func = PMLoss(model_renderer,
                            loss_func='L1',
                            translation_range=train_dataset.metadata['translation_range'],
                            rotation_range=train_dataset.metadata['rotation_range'])
    else:
        loss_func = torch.nn.MSELoss()
    # model = model_class(image_size=int(train_dataset.metadata["image_size"]), phase=phase, fx=camera.focal_x, fy=camera.focal_y, delta_pose=delta_pose, loss_func=loss_func)
    
    if architecture in ["res_vit_single"]:
        model = model_class(image_size=int(train_dataset.metadata["image_size"]), phase=phase, loss_func=loss_func, no_token=no_token, hybrid_vit=hybrid_vit, more_heads=more_heads, smaller=smaller)
    elif architecture in ["trans1", "trans2", "trans2hyb", "trans2single", "trans2singlehyb", "res_vit", "res_vit_single"]:
        model = model_class(image_size=int(train_dataset.metadata["image_size"]), phase=phase, loss_func=loss_func, no_token=no_token, hybrid_vit=hybrid_vit, more_heads=more_heads)
    elif architecture in ["fpnet", "fplike4", "fplike7", "fplike8", "fplike9", "fplike10", "fplike11", "fplike12"]:
        model = model_class(cfg={'use_BN': True, 'rot_rep': 'axis_angle'}, loss_func=loss_func)
    else:
        model = model_class(image_size=int(train_dataset.metadata["image_size"]), phase=phase, loss_func=loss_func)
    if finetune_path != "None":
        finetune_path = os.path.expandvars(finetune_path)
        print("Finetuning path : {}".format(finetune_path))
        checkpoint = torch.load(finetune_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        if phase == 1:
            model.load_hal()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    #scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 15], gamma=0.1)
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


    # Instantiate the train loop and train the model.
    train_loop_handler = TrainLoop(model, train_loader, val_loader, optimizer, backend, gradient_clip, scheduler=scheduler)
    train_loop_handler.add_callback(callbacks)
    print("Training Begins:")
    train_loop_handler.loop(epochs, output_path, load_best_checkpoint=start_from_last, save_all_checkpoints=False)
    print("Training Complete")
