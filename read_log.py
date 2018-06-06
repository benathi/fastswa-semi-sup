import matplotlib
matplotlib.use("Agg")
import msgpack
from pandas import DataFrame
import pandas
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import math

def filter_name(agname):
  name = None
  if agname == 'validation':
    name = 'student'
  elif agname == 'ema_validation':
    name = 'ema'
  elif agname == 'swa_validation':
    name = 'swa'
  elif 'fastswa_validation_freq' in agname and agname[len("fastswa_validation_freq"):].isdigit():
    digit = int(agname[len("fastswa_validation_freq"):])
    name = 'fastswa-freq-{}'.format(digit)
  else:
    print("Not Supported")
    name = agname
  print("{} -> {}".format(agname, name))
  return name

def plot_prec(dirpath, figname='test.png', cutoff=90.0, verbose=False, interval=1.0, upper=100, forceplot=False):
  assert os.path.isdir(dirpath)
  if not os.path.isfile(os.path.join(dirpath, 'validation.msgpack')):
    return
  fig, ax = plt.subplots()
  ax2 = ax.twinx()
  start_zero = True
  _types = ['validation', 'ema_validation', 'swa_validation'] +  \
    ['fastswa_validation_freq3', '', 'fastswa_validation_freq20']
  for _type in _types:
    logpath = os.path.join(dirpath, _type + '.msgpack')
    if not os.path.isfile(logpath):
      continue
    _df = pandas.read_msgpack(logpath)
    if verbose:
      print("type=", _type)
      print(list(_df.columns.values))

    if sum(list(_df['top1/avg'] > cutoff)) == 0:
      print("Not plotting", _type)
      continue
    cutoff_idx = np.argmax(np.array(list(_df['top1/avg'] > cutoff)))

    for col in ['top1/avg']:
      if _df['step'].iloc[0] > 1:
        start_zero = False
      agname = filter_name(_type)
      steps = _df['step'].iloc[cutoff_idx:].index
      if len(steps) < 2 and 'swa' not in _type:
        print("Not enough steps to plot")
        if verbose: print(steps)
        if not forceplot: return
      vals = _df[col].iloc[cutoff_idx:]
      ax.plot(steps, vals, label=agname)
      ax2.plot(steps, vals, label=agname)

  ax.set_yticks(np.arange(cutoff, upper + interval, interval))
  ax2.set_yticks(np.arange(cutoff, upper + interval/2, interval/2))
  ax2.grid(color='b', linestyle='-', linewidth=1, alpha=0.2)

  ax.set_xlabel("Epoch")
  ax.set_ylabel("Accuracy")
  ax2.set_ylabel("Accuracy")

  plt.legend()
  print("Saving figure to", figname)
  fig.savefig(os.path.join("figs", figname))

def plot_batch(pattern, verbose=False, cutoff=85, interval=1., upper=100, forceplot=False):
  listdirs = glob.glob(pattern, recursive=True)
  if verbose: print(listdirs)
  EXPNAME=1
  DATE=2
  NSEED=3
  for dirname in listdirs:
    dirname = dirname.split('/')
    n, seed = dirname[NSEED].split('_')
    expname = dirname[EXPNAME]
    date = dirname[DATE]
    dirpath = "/".join(dirname[:-1])
    print('dirpath=', dirpath)
    figname = "N-{}_seed-{}_exp-{}_date-{}.png".format(n, seed, expname, date)
    plot_prec(dirpath=dirpath, figname=figname, cutoff=cutoff, verbose=verbose, interval=interval, upper=upper, forceplot=forceplot)

if __name__=='__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--pattern', default='results/cifar10*', type=str)
  parser.add_argument('--cutoff', default=85, type=int)
  parser.add_argument('--interval', default=1, type=int)
  parser.add_argument('--verbose', default=0, type=int)
  parser.add_argument('--upper', default=100, type=int)
  parser.add_argument('--forceplot', default=0, type=int)
  args = parser.parse_args()
  plot_batch(pattern=args.pattern + "*/**/transient", cutoff=args.cutoff, interval=args.interval,
    upper=args.upper,
    verbose=args.verbose,
    forceplot=args.forceplot
    )








