#!/usr/bin/env bash
pip install tqdm; pip install matplotlib; pip install pandas; pip install msgpack
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Downloading and unpacking CIFAR-10"
mkdir -p $DIR/../workdir
#mkdir -p $DIR/../images/cifar/cifar10/by-image/ # not needed
python $DIR/unpack_cifar10.py $DIR/../workdir $DIR/../images/cifar/cifar10/by-image/

echo "Linking training set"
(
    cd $DIR/../images/cifar/cifar10/by-image/
    bash $DIR/link_cifar10_train.sh
)

echo "Linking validation set"
(
    cd $DIR/../images/cifar/cifar10/by-image/
    bash $DIR/link_cifar10_val.sh
)
# ./data-local/bin/prepare_cifar10.sh: line 9: 0/by-image/: No such file or directory   -> is this a problem?