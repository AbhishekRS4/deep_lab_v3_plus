# DeepLab\_v3+ implementation on Cityscapes dataset

## Notes
* Implementation of DeepLab\_v3+ with ResNet-50
* The original implementation uses Xception pretrained encoder 
* The image dimension used to train the model is 1024x512
* 15 custom classes used

## Main idea
* Apply atrous spatial pyramid pooling to feature maps of output stride 16 of the input size. Perform bilinear upsampling by a factor of 4 and concatenate corresponding encoder stage features and perform bilinear upsampling by a factor of 4 again.

## Intructions to run
> To run training use - **python3 deep\_lab\_v3\_plus\_train.py -h**
>
> To run inference use - **python3 deep\_lab\_v3\_plus\_infer.py -h**
>
> This lists all possible commandline arguments

## Visualization of results
* [DeepLabv3+]()

## Reference
* [ResNet-50](https://arxiv.org/abs/1512.03385)
* [DeepLabv3+](https://arxiv.org/pdf/1802.02611.pdf)
* [Cityscapes Dataset](https://www.cityscapes-dataset.com/)

## To do
- [x] DeepLabv3+
- [ ] Visualize results
- [ ] Compute metrics
