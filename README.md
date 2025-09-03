# DeepTrack++

Upgrades to the 6DoF object tracker presented in : "A Framework for Evaluating 6-DOF Object Trackers" [\[arxiv paper\]](https://arxiv.org/abs/1803.10075).

The original code release can be found at this [website](https://github.com/lvsn/6DOF_tracking_evaluation).

The dataset can be downloaded at this [website](http://vision.gel.ulaval.ca/~jflalonde/publications/projects/6dofObjectTracking/index.html).

## Changes to DeepTrack

The ``bb3d`` parameter is used for depth values clipping, detailed in Charles Renaud's thesis (under review).

The ``newbg3`` parameter is used for the background values augmentation scheme, detailed in Charles Renaud's thesis (under review).

Includes the `fpnet` architecture from [FoundationPose](https://github.com/NVlabs/FoundationPose).

## Dependencies
To train the network, version 0.1 of [pytorch_toolbox](https://github.com/MathGaron/pytorch_toolbox/tree/v0.1) is required.

Other dependencies are listed in `requirements.txt`

## Citation

If you use this dataset in your research, please cite:
```
@inproceedings{garon2018framework,
	       title={A framework for evaluating 6-dof object trackers},
	       author={Garon, Mathieu and Laurendeau, Denis and Lalonde, Jean-Fran{\c{c}}ois},
	       booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
	       pages={582--597},
	       year={2018}
}
```

# Tracker
## Generate the dataset
Change the parameters in ``generator_script.sh`` and run to generate the training and validation dataset.

## Train the network
Change the parameters in ``train_script.sh`` and run to train the network.

# License

```
License for Non-Commercial Use

If this software is redistributed, this license must be included.
The term software includes any source files, documentation, executables,
models, and data.

This software is available for general use by academic or non-profit,
or government-sponsored researchers. This license does not grant the
right to use this software or any derivation of it for commercial activities.
For commercial use, please contact Jean-Francois Lalonde at Université Laval
at jflalonde@gel.ulaval.ca.

This software comes with no warranty or guarantee of any kind. By using this
software, the user accepts full liability.
```