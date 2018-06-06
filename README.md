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

We provide training scripts in folder *exps*. To replicate the results for CIFAR-10 with 4000 labels using CNN, run the following:

```
python experiments/cifar10_mt_cnn_short_n4k.py
python experiments/cifar10_pi_cnn_short_n4k.py
```

Similarly, for CIFAR-100 with 10k labels:
```
python experiments/cifar100_mt_cnn_short_n10k.py
```


Note: the code is adapted from https://github.com/CuriousAI/mean-teacher/tree/master/pytorch