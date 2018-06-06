import sys
import logging
import torch
sys.path.append('.')
import main
from mean_teacher.cli import parse_dict_args
from mean_teacher.run_context import RunContext

LOG = logging.getLogger('runner')


def parameters():
    defaults = {
        # Technical details
        'workers': 2,
        'checkpoint_epochs': 2,

        # Data
        'dataset': 'cifar100',
        'train_subdir': 'train+val',
        'eval_subdir': 'test',

        # Data sampling
        'base_batch_size': 128,
        'base_labeled_batch_size': 31,

        # Architecture
        'arch': 'cifar_shakeshake26',
        'ema_decay': 0.97,

        # Costs
        'consistency_type': 'mse',
        'consistency_rampup': 5,
        'consistency': 100.0,
        'logit_distance_cost': .01,
        'weight_decay': 2e-4,

        # Optimization
        'epochs': 180,
        'lr_rampup': 0,
        'base_lr': 0.1,
        'lr_rampdown_epochs': 210,
        'nesterov': True,

        'num_cycles': 20,
        'cycle_interval': 30,
        'start_epoch': 0,
        'swa-frequencies': '3',
        'unsup_augment': 'tiny_237k',
        'limit_unlabeled': 100000,
    }

    for n_labels  in [50000]:
        for data_seed in [10]:
            yield {
                **defaults,
                'title': '{}-label cifar-100'.format(n_labels),
                'n_labels': n_labels,
                'data_seed': data_seed
            }


def run(title, base_batch_size, base_labeled_batch_size, base_lr, n_labels, data_seed, **kwargs):
    LOG.info('run title: %s', title)
    ngpu = torch.cuda.device_count()
    assert ngpu == 1
    adapted_args = {
        'batch_size': base_batch_size * ngpu,
        'labeled_batch_size': base_labeled_batch_size * ngpu,
        'lr': base_lr * ngpu,
        'labels': 'data-local/labels/cifar100/{}_balanced_labels/{:02d}.txt'.format(n_labels, data_seed),
    }
    context = RunContext(__file__, "{}_{}".format(n_labels, data_seed))
    main.args = parse_dict_args(**adapted_args, **kwargs)
    main.main(context)


if __name__ == "__main__":
    for run_params in parameters():
        run(**run_params)
