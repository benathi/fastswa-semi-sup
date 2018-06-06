#!/usr/bin/env bash

# Prepare semisupervised datasets needed in the experiments

SCRIPT=./labels/bin/create_balanced_semisupervised_labels_trainval.sh

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

DATADIR=images/cifar/cifar10/by-image/
DATANAME=cifar10
NUM_CLASSES=10
LABEL_VARIATIONS=(5000)
create