#!/usr/bin/env bash

# Prepare semisupervised datasets needed in the experiments
# usage: at data-local directory, run the following:
# ./labels/bin/prepare_semisupervised_sets_cifar100.sh
SCRIPT=./labels/bin/create_balanced_semisupervised_labels.sh

create ()
{
    for LABELS_PER_CLASS in ${LABEL_VARIATIONS[@]}
    do
        LABELS_IN_TOTAL=$(( $LABELS_PER_CLASS * $NUM_CLASSES ))
        echo "Creating sets for $DATANAME with $LABELS_IN_TOTAL labels."
        for IDX in {00..19}
        do
            LABEL_DIR=labels/${DATANAME}/${LABELS_IN_TOTAL}_balanced_labels
            mkdir -p $LABEL_DIR
            $SCRIPT $DATADIR $LABELS_PER_CLASS > $LABEL_DIR/${IDX}.txt
        done
    done
}

DATADIR=images/cifar/cifar100/by-image
DATANAME=cifar100
NUM_CLASSES=100
LABEL_VARIATIONS=(10 40 100 200)
create