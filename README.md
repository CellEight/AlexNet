# AlexNet

## Background

This is a Pytorch implementation of the famous paper ["ImageNet Classification with Deep Convolutional Neural Networks"](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) by Alex Krizhevsky, Ilya Sutskever and Geoffrey E. Hinton or, as the architecture described within is more commonly know, AlexNet.

## The Architecture

![The AlexNet Architecture](./model.png)

## Deviations From the Paper

I have remained as faithful as is reasonable to the original paper however it is no longer 2012.
In the original paper the authors had to contend with the limitations of contemporary graphics cards such as the GTX 580 which had a measly 3GB of graphics memory and to do this they opted to implement a parallelization scheme where they split the model across two separate GPUs with the two communicating only on certain layers.
It would have been somewhat foolish to recreate this "hack" they implemented to cope with limitations that are no longer extant (The RTX 2070 in my laptop alone has 8GB of graphics RAM!) so as a result in my implementation I have opted to remove it.
Rather than having two separate, parallel network paths I have merged the corresponding layers into singular layers of the equivalent size.
I apologize if you were in search of an implementation of this particular part of their architecture, you shall not find it here!

I am also yet to implement local response normalization which is a additional feature of the architecture described in section 3.3 of the paper.
The authors noted a around 2% improvement of the models performance when trained on CIFAR-10 when using this technique but despite this limited gain in performance I still plan to add it into the code in the near future.

## Dataset

In the paper the authors train the model on the ImageNet dataset which is a huge data set of images of objects in 1000 different classes and was used as part of the ImageNet competition which until recently was the main forum of comparison between state of the art image recognition models.
Although this model is absolutely capable of being applied to the full image net dataset I do not recommend this as it is VERY large, approximately 138GB.
Instead, if you wish to train the model yourself, I recommend either using the sample data set in the `/data` directory of the repo which is just a small subset of 11 classes taken from the ImageNet data set or downloading your own subset using the [ImageNet Downloader](https://github.com/mf1024/ImageNet-Datasets-Downloader) project and then using imagemagick and the converter script included in this repo to get the images the correct size and training with that.
If you really want to train with all of ImageNet you can find a few different methods of acquiring it [here](http://www.cloverio.com/download-imagenet/).

## Pretrained Weights

If you lack a graphics card on which to train the model or you just don't want to go through the hassle of training it yourself I have uploaded a `.pkl` file containing a serialized version of the model trained on the included dataset.
You can find it [here](https://drive.google.com/file/d/1YKlLTGwb3yzXBqGW1FcbAZjbcsZvddEC/view?usp=sharing).

## Requirements

All you need to run this code are the torch and torchvision libraries.
To install these just run the following command in the root of your local copy of the repo.
Do bear in mind though that you may wish to visit the pytorch website to download the most appropriate versions for your system.
```
sudo pip3 install -r ./requirements.txt
```
