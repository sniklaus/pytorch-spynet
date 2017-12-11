# pytorch-spynet
This is a personal reimplementation of Optical Flow Estimation using a Spatial Pyramid Network [1] using PyTorch. Should you be making use of this work, please cite the paper accordingly. Also, make sure to adhere to the <a href="https://github.com/anuragranj/spynet#license">licensing terms</a> of the authors. Should you be making use of this particular implementation, please acknowledge it appropriately.

For the original Torch version of this work, please see: https://github.com/sniklaus/torch-sepconv

## setup
To download the pre-trained networks, run `bash install.bash` and make sure that you have Torch installed. The pre-trained networks are obtained from the original repository of the authors and are converted using a Torch script. While it would be nice to provide the converted weights myself and thus being able to avoid the initial Torch dependency, I do not know whether I would have the rights to do so. However, Torch is no longer required after the conversion.

## usage
To run it on your own pair of images, use the following command. You can choose between various models, please make sure to see their paper / the code for more details.

```
python run.py --model sintel-final --first ./images/first.png --second ./images/second.png --out ./out.flo
```

## license
As stated in the <a href="https://github.com/anuragranj/spynet#license">licensing terms</a> of the authors of the paper, the models are free for non-commercial and scientific research purpose. Please make sure to further consult their licensing terms.

## references
```
[1]  @inproceedings{Ranjan_CVPR_2017,
         author = {Ranjan, Anurag and Black, Michael J.},
         title = {Optical Flow Estimation Using a Spatial Pyramid Network},
         booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
         year = {2017}
     }
```