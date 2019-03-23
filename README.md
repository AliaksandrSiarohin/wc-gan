# Whitening and Coloring transform for GANs

[Check out our paper](https://arxiv.org/abs/1806.00420)
<p>
  <table>
	<tr>
           <td> <img src="sup-mat/cifar10_SN_uncond.png" width="350"/> <figcaption align="center">Unsupervised Cifar-10 (IS 8.66)</figcaption> </td>
           <td> <img src="sup-mat/cifar10_SN_cls.png" width="350"/> <figcaption align="center">Supervised Cifar-10 (IS 9.06)</figcaption> </td>
        </tr>
  </table> 
</p>

### Requirments
* python2
* numpy
* scipy
* skimage
* pandas
* tensorflow == 1.5.0 (I have not testted with other versions)
* keras == 2.0.8 (I have tried latter versions, but they throw a bug. Not sure from where it came from)
* tqdm 


For the commands reproducing experiments from the paper check scripts folder.

All scripts has the following name: (name of the dataset) + (architecture type (resnet or dcgan)) +
(discriminator normalization (sn or wgan_gp)) + (conditional of unconditional) + (if conditional use soft assigment (sa)).

For example:

```CUDA_VISIBLE_DEVICES=0 scripts/cifar10_resnet_sn_uncond.sh```

will train GAN for cifar10 dataset, with resnet architecture, spectral normalized discriminator in unconditional case.


All dataset except for imagenet downloaded and trained at the same time.

### Imagenet

0. This will consume a loot of memory. Because dataset is packed into numpy files for sequential reads.
1. Download imagenet [ILSVRC2012](http://image-net.org/download-images). Train and val. Put train to ../ILSVRC2012/train, and val to ../ILSVRC2012/val/val (val/val is important)
2. Preprocess  imagenet train:

```bash preprocess.sh ../ILSVRC2012/train ../imagenet-resized```

3. Preprocess imagenet val:

```bash preprocess.sh ../ILSVRC2012/val ../imagenet-resized-val```

4. Now you can remove ILSVRC2012
5. ```CUDA_VISIBLE_DEVICES=0 scripts/imagenet_resnet_sn_cond_sa.sh``` This will first pack imagenet into numpy files, and then start traning.

Citation:
```
@inproceedings{
	siarohin2018whitening,
	title={Whitening and Coloring transform for {GAN}s},
	author={Aliaksandr Siarohin and Enver Sangineto and Nicu Sebe},
	booktitle={ICLR},
	year={2019},
	url={https://openreview.net/forum?id=S1x2Fj0qKQ}
}
```


