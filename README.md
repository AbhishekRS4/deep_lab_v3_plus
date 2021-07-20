# DeepLab\_v3+ implementation on Cityscapes dataset

## Notes
* Implementation of DeepLab\_v3+ with ResNet-50
* The original implementation uses Xception pretrained encoder
* The image dimension used to train the model is 1024x512
* 15 custom classes used

## Main idea
* Apply atrous spatial pyramid pooling to feature maps of output stride 16 of the input size. Perform bilinear upsampling by a factor of 4 and concatenate corresponding encoder stage features and perform bilinear upsampling by a factor of 4 again.

## Intructions to run
* To list training options
```
python3 src/deep_lab_v3_plus_train.py --help
```
* To list inference options
```
python3 src/deep_lab_v3_plus_infer.py --help
```

## Visualization of results
* [DeepLabv3+]()

## Reference
* [ResNet-50](https://arxiv.org/abs/1512.03385)
* [DeepLabv3+](https://arxiv.org/pdf/1802.02611.pdf)
* [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
