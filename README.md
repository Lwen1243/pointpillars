# Contents

- [Contents](#contents)
    - [PointPillars description](#pointpillars-description)
    - [Model architecture](#model-architecture)
    - [Dataset](#dataset)

## [PointPillars description](#contents)

PointPillars is a method for object detection in 3D that enables end-to-end learning with only 2D convolutional layers.
PointPillars uses a novel encoder that learn features on pillars (vertical columns) of the point cloud to predict 3D oriented boxes for objects.
There are several advantages of this approach.
First, by learning features instead of relying on fixed encoders, PointPillars can leverage the full information represented by the point cloud.
Further, by operating on pillars instead of voxels there is no need to tune the binning of the vertical direction by hand.
Finally, pillars are highly efficient because all key operations can be formulated as 2D convolutions which are extremely efficient to compute on a GPU.
An additional benefit of learning features is that PointPillars requires no hand-tuning to use different point cloud configurations.
For example, it can easily incorporate multiple lidar scans, or even radar point clouds.

> [Paper](https://arxiv.org/abs/1812.05784):  PointPillars: Fast Encoders for Object Detection from Point Clouds.
> Alex H. Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, Oscar Beijbom, 2018.

## [Model architecture](#contents)

The main components of the network are a Pillar Feature Network, Backbone, and SSD detection head.
The raw point cloud is converted to a stacked pillar tensor and pillar index tensor.
The encoder uses the stacked pillars to learn a set of features that can be scattered back to a 2D pseudo-image for a convolutional neural network.
The features from the backbone are used by the detection head to predict 3D bounding boxes for objects.

## [Dataset](#contents)

Dataset used: [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)

Data was collected with using a standard station wagon with two high-resolution color and grayscale video cameras.
Accurate ground truth is provided by a Velodyne laser scanner and a GPS localization system.
Dataset was captured by driving around the mid-size city of Karlsruhe, in rural areas and on highways.
Up to 15 cars and 30 pedestrians are visible per image. The 3D object detection benchmark consists of 7481 images.


## Why to do this

Pointpillars is hard to convert to rk3588 and other platform, so I try to rebuild it

## How to use it

please refer to https://github.com/LKLQQ/pointpillars to get more information