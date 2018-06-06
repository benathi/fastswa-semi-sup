# fastswa-semi-sup
Improving Consistency-Based Semi-Supervised Learning with Weight Averaging 


The code runs on Python 3 with Pytorch 0.3. The following packages are also required.
```
pip install scipy tqdm matplotlib pandas msgpack
```

Then prepare CIFAR-10 and CIFAR-100 with the following commands:

```
./data-local/bin/prepare_cifar10.sh
./data-local/bin/prepare_cifar100.sh
```

We provide training scripts in folder *exps*. To replicate the results for CIFAR-10 using the Mean Teacher model + fast-SWA on 4000 labels with a 13-layer CNN, run the following:

```
python experiments/cifar10_mt_cnn_short_n4k.py
```

Similarly, for CIFAR-100 with 10k labels:
```
python experiments/cifar100_mt_cnn_short_n10k.py
```

<p align="center"> 
<img src="figs/N-4000_seed-10_exp-cifar10_mt_cnn_short_n4k_date-2018-06-06_01-40-29.png">

Figure 1. CIFAR-10 Test Accuracy of the Mean Teacher Model with SWA and fastSWA using a 13-layer CNN and 4000 labels.
</p>

<p align="center"> 
<img src="figs/N-10000_seed-11_exp-cifar100_mt_cnn_short_n10k_date-2018-06-06_01-45-02.png">

Figure 2. CIFAR-100 Test Accuracy of the Mean Teacher Model with SWA and fastSWA using a 13-layer CNN and 10000 labels.
</p>

We provide scripts for ResNet-26 with Shake-Shake regularization for CIFAR-10 and CIFAR-100, as well as other label settings in the directory **experiments**.


Note: the code is adapted from https://github.com/CuriousAI/mean-teacher/tree/master/pytorch
