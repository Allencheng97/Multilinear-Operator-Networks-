Multilinear Operator Networks
===================================================

This is a PyTorch implementation of the ICLR'24 paper [Multilinear Operator Networks](https://arxiv.org/abs/2401.17992).



## Usage


### Install

- PyTorch version: 1.13.1 + and CUDA version: 11.7
- timm version: 0.9.13dev0
- einops and fvcore
  
You can also install through
```
pip install -r requirements.txt
```

### Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is:

```
│path/to/imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

### Evaluation

To evaluate a pre-trained MONet-S on ImageNet val with a single GPU run:
```
python validate.py  [imagenet_val folder] --model MONet_S   --checkpoint [ckpt_path]  -b 256
```
You could download the checkpoint through [here](https://drive.google.com/file/d/1OsAS3pD4LrfsTmiy69S4wvCF9Mh4PBd0/view?usp=sharing))
### Training

To train MONet on ImageNet on a single node with n GPUs for 300 epochs run:
You should first change the CUDA_VISIBLE_DEVICES index in ./distributed_train.sh
Then run
```
./distributed_train.sh n [ImageNet Folder] --img-size 224 --model [Model name] --num-classes 1000 --epochs 300 --opt adamw --clip-grad 1 --batch-size [batch size] --weight-decay [wd] --sched cosine --lr [lr]  [DataAugment Recipe] 
```
Please refer to timm repo and our paper for [DataAugment Recipe] parameters

You could also add --amp in the end to enable automatic mixed precision to save VRAM

### Acknowledgement

This code is based on [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) and  [Jitter MLP](https://github.com/liuruiyang98/Jittor-MLP). Thanks for their wonderful works

## Citing

```bibtex
@inproceedings{cheng2024multilinear,
  title={Multilinear Operator Networks},
  author={Cheng, Yixin and Chrysos, Grigorios G and Georgopoulos, Markos and Cevher, Volkan},
   booktitle={International Conference on Learning Representations},
  year={2024}
}
```


