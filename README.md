# pytorch-spynet
This is a personal reimplementation of SPyNet [1] using PyTorch. Should you be making use of this work, please cite the paper accordingly. Also, make sure to adhere to the <a href="https://github.com/anuragranj/spynet#license">licensing terms</a> of the authors. Should you be making use of this particular implementation, please acknowledge it appropriately [2].

<a href="https://arxiv.org/abs/1611.00850" rel="Paper"><img src="http://www.arxiv-sanity.com/static/thumbs/1611.00850v1.pdf.jpg" alt="Paper" width="100%"></a>

For the original Torch version of this work, please see: https://github.com/anuragranj/spynet
<br />
Another optical flow implementation from me: https://github.com/sniklaus/pytorch-pwc
<br />
And another optical flow implementation from me: https://github.com/sniklaus/pytorch-liteflownet
<br />
Yet another optical flow implementation from me: https://github.com/sniklaus/pytorch-unflow

## setup
To download the pre-trained models, run `bash download.bash`. These originate from the original authors, I just converted them to PyTorch.

## usage
To run it on your own pair of images, use the following command. You can choose between various models, please make sure to see their paper / the code for more details.

```
python run.py --model sintel-final --first ./images/first.png --second ./images/second.png --out ./out.flo
```

I am afraid that I cannot guarantee that this reimplementation is correct. However, it produced results identical to the implementation of the original authors in the examples that I tried. Please feel free to contribute to this repository by submitting issues and pull requests.

## comparison
<p align="center"><img src="comparison/comparison.gif?raw=true" alt="Comparison"></p>

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

```
[2]  @misc{pytorch-spynet,
         author = {Simon Niklaus},
         title = {A Reimplementation of {SPyNet} Using {PyTorch}},
         year = {2018},
         howpublished = {\url{https://github.com/sniklaus/pytorch-spynet}}
    }
```