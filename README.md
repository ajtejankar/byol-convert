# Convert official BYOL weights to PyTorch

Only supports ResNet-50 for now. Since the augmentation in PyTorch will be slightly different from the orignal, expect some differences in accuracy. I am not entirely sure about why the `crop_only` is 5 point worse. Original weights from [BYOL](https://github.com/deepmind/deepmind-research/tree/master/byol). This is a basic script which should generally work with most versions of `PyTorch` and `Torchvision`, but it's written with `PyTorch (1.4)` and `Torchvision (0.5)`.

```
# convert the weights
python convert.py pretrain_res50x1.pkl pretrain_res50x1.pth.tar
# validate the weights
python validate.py pretrain_res50x1.pth.tar /datasets/imagenet/val
```



| Name | Original Acc | Converted Acc |
| ----------- | ----------- | ----------- |
| pretrain_res50x1 | 74.4 | [74.6](https://drive.google.com/file/d/1nwaOpgmjpiOxJez7gUKQmYEiQIJe5Yss/view?usp=sharing) |
| res50x1_batchsize_2048 | 72.4 | 72.3 |
| res50x1_batchsize_1024 | 72.2 | 72.3 |
| res50x1_batchsize_512 | 72.2 | 72.1 |
| res50x1_batchsize_256 | 71.8 | 71.9 |
| res50x1_batchsize_128 | 69.6 (+- 0.5) | 69.7 |
| res50x1_batchsize_64 | 59.7 (+- 1.5) | 58.2 |
| res50x1_crop_and_blur_only | 61.1 (+- 0.3) | 62.9 |
| res50x1_crop_and_color_only | 70.7 | 69.1 |
| res50x1_crop_only | 59.4 (+- 0.3) | 55.3 |
| res50x1_no_color | 63.4 (+- 0.7) | 63.8 |
| res50x1_no_grayscale | 70.3 | 70.5 |
