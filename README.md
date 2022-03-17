# DeepLadder

This is a Tensorflow implementation for
* [DeepLadder](https://dl.acm.org/doi/abs/10.1145/3458306.3458873): Deep Reinforced Bitrate Ladders for Adaptive Video Streaming 

## Cite

If you find this work useful to you, please cite the [conference version](https://dl.acm.org/doi/abs/10.1145/3458306.3458873):

```
@inproceedings{huang2021deep,
  title={Deep reinforced bitrate ladders for adaptive video streaming},
  author={Huang, Tianchi and Zhang, Rui-Xiao and Sun, Lifeng},
  booktitle={Proceedings of the 31st ACM Workshop on Network and Operating Systems Support for Digital Audio and Video},
  pages={66--73},
  year={2021}
}
```

## Quick Start

We provide all video & trace datasets.

For CBR videos, we list:

```
deepladder-cbr/feature: the feature maps from the last convolution layer (ResNet50) for each video.
deepladder-cbr/sizes: the video sizes for each video, note, encoded with different bitrates.
deepladder-cbr/vmaf: video quality (VMAF) with different videos and encoding bitrates.
```

For VBR videos, we list:

```
deepladder-vbr/train/ssim: video quality feature (SSIM)
deepladder-vbr/train/feature: feature map
deepladder-vbr/train/size: video sizes
```

Meanwhile, we give two imeplementations, i.e., DeepLadder-CBR and DeepLadder-VBR. Type

```
python train.py
```

to train DeepLadder.

## Contact

This work was done about one year ago, and we are not sure whether the current version is the lastest version or not.
So feel free to let us know if you have any questions.
