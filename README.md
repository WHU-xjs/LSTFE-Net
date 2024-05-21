# LSTFE-Net:Long Short-Term Feature Enhancement Network for Video Small Object Detection
This repo is an official implementation of ["LSTFE-net: Long short-term feature enhancement network for video small object detection"](http://openaccess.thecvf.com/content/CVPR2023/papers/Xiao_LSTFE-NetLong_Short-Term_Feature_Enhancement_Network_for_Video_Small_Object_Detection_CVPR_2023_paper.pdf), accepted by CVPR 2023. 

## Citing LSTFE
Please cite our paper in your publications if it helps your research:
```
@inproceedings{xiao2023lstfe,
  title={LSTFE-net: Long short-term feature enhancement network for video small object detection},
  author={Xiao, Jinsheng and Wu, Yuanxu and Chen, Yunhua and Wang, Shurui and Wang, Zhongyuan and Ma, Jiayi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14613--14622},
  year={2023}
}
```
## Installation

Please follow [INSTALL.md](INSTALL.md) for installation instructions.


## Data preparation

Please download FLDrones dataset. After that, we recommend to symlink the path to the datasets to `datasets/`. And the path structure should be as follows:

    ./datasets/FLDrones/
    ./datasets/FLDrones/Annotations/VID
    ./datasets/FLDrones/Data/VID
    ./datasets/FLDrones/ImageSets

### Inference

The inference command line for testing on the validation dataset:

    python -m torch.distributed.run \
        --nproc_per_node 1 \
        tools/test_net.py \
        --config-file configs/LSTFE/vid_R_101_C4_LSTFE_1x.yaml \
        MODEL.WEIGHT FLDrones_lstfe.pth 
        
Please note that:
1) `FLdrones_lstfe.pth` is your model name
2) If you want to evaluate a different model, please change `--config-file` to its config file and `MODEL.WEIGHT` to its weights file.

### Training

The following command line will train LSTFE_Resnet101 on 1 GPUs with Synchronous Stochastic Gradient Descent (SGD):

    python -m torch.distributed.run \
        --nproc_per_node 1 \
        tools/train_net.py \
        --config-file configs/LSTFE/vid_R_101_C4_LSTFE_1x.yaml \
        OUTPUT_DIR training_dir/LSTFE
        
Please note that:
1) The models will be saved into `OUTPUT_DIR`.

## Contributing to the project
Any pull requests or issues are welcomed.