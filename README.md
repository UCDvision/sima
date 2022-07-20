# Simple Softmax-free Attention for Vision Transformer (SimA)


Official PyTorch implementation and pretrained models for SimA models. ([arXiv](https://arxiv.org/abs/2206.08898)) 

---

```
@misc{https://doi.org/10.48550/arxiv.2206.08898,
  doi = {10.48550/ARXIV.2206.08898},
  url = {https://arxiv.org/abs/2206.08898},
  author = {Koohpayegani, Soroush Abbasi and Pirsiavash, Hamed},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {SimA: Simple Softmax-free Attention for Vision Transformers},
  publisher = {arXiv},
  year = {2022},  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

# Getting Started


You can install the required packages including: [Pytorch](https://pytorch.org/) version 1.7.1, [torchvision](https://pytorch.org/vision/stable/index.html) version 0.8.2 and [Timm](https://github.com/rwightman/pytorch-image-models) version 0.4.8
```
pip install -r requirements.txt
```

Download and extract the [ImageNet](https://imagenet.stanford.edu/) dataset. Afterwards, set the ```--data-path``` argument to the corresponding extracted ImageNet path.





### Training

For training using 8 gpus, use the following command

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model sima_small_12_p16 --epochs 400 --batch-size 128 --drop-path 0.05 --output_dir [OUTPUT_PATH]   --data-path [DATA_PATH]
```
