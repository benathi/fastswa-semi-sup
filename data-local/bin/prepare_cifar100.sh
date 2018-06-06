#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Downloading and unpacking CIFAR-100"
mkdir -p $DIR/../workdir
python $DIR/unpack_cifar100.py $DIR/../workdir $DIR/../images/cifar/cifar100/by-image/

echo "Linking training set"
(
    cd $DIR/../images/cifar/cifar100/by-image/
    bash $DIR/link_cifar100_train.sh
)

echo "Linking validation set"
(
    cd $DIR/../images/cifar/cifar100/by-image/
    bash $DIR/link_cifar100_val.sh
)
