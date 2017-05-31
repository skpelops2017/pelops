
# coding: utf-8

# In[ ]:

import os
import re
import sys
import glob
import pprint
import pickle
import collections
import multiprocessing

pelops_dir = os.path.abspath('../..')
sys.path.insert(0, pelops_dir)

from pelops.datasets.featuredataset import FeatureDataset
from pelops.experiment_api.experiment import ExperimentGenerator
from pelops.analysis import analysis
import matplotlib.pyplot as plt


# In[ ]:

# Set experiment constants

ITEMS_PER_CAMERA = 10
Y_RANDOM=1024
CAMERAS=2
DROPPED=0
CMC_CNT=100
EXPERIMENTS=400

# Set data constants
DATA_ROOT = os.path.expanduser('~/Data')
OUT_DIR = os.path.expanduser('~/Results')


# In[ ]:

# Scan for and locate feature datasets

files_by_network = collections.defaultdict(dict)
files_by_dataset = collections.defaultdict(dict)
file_pattern = re.compile('^(?P<key>.+?)_\d+px_[A-Za-z]+\.hdf5', re.S | re.I)

for root, sub_dirs, file_names in os.walk(DATA_ROOT):
    full_names = [os.path.join(root, fn) for fn in file_names if fn.endswith('.hdf5')]
    if len(full_names) > 0:
        dataset_name = os.path.basename(root)
        for file_name in full_names:
            mch = re.search(file_pattern, os.path.basename(file_name))
            if mch is None:
                print('ERROR: {}'.format(file_name))
            network_name = mch.group('key')
            files_by_network[network_name][dataset_name] = file_name
            files_by_dataset[dataset_name][network_name] = file_name
            
formatter = lambda d, s: '{}:\n{}'.format(s, pprint.pformat({k: [h for h in v] for k, v in d.items()}))
# print(formatter(files_by_network, 'Files By Network'))
# print(formatter(files_by_dataset, 'Files By Dataset'))


# In[ ]:

# Define experiment behavior and processing output

def run_experiment(args):
    feature_file, title, output_file = args
    print(title)
    
    # Require output dir
    out_dir = os.path.dirname(output_file)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    # Run experiment
    features = FeatureDataset(feature_file)
    experiment_gen = ExperimentGenerator(features, CAMERAS, ITEMS_PER_CAMERA, DROPPED, Y_RANDOM)
    experiment_hldr = analysis.repeat_pre_cmc(features, experiment_gen, NUMCMC=CMC_CNT, EXPPERCMC=EXPERIMENTS)
    stats, gdata = analysis.make_cmc_stats(experiment_hldr, ITEMS_PER_CAMERA)
    
    # Save raw experiment results
    meta = {
        'feature_file': feature_file,
        'title': title,
        'output_file': output_file,
        'stats': stats,
        'gdata': gdata,
        'ITEMS_PER_CAMERA': ITEMS_PER_CAMERA,
        'Y_RANDOM': Y_RANDOM,
        'CAMERAS': CAMERAS,
        'DROPPED': DROPPED,
        'CMC_CNT': CMC_CNT,
        'EXPERIMENTS': EXPERIMENTS
    }
    with open(output_file + '.pkl', 'wb') as out_hdl:
        pickle.dump(meta, out_hdl)

    # Plot experiment results
    figure = plt.figure()
    ax = plt.subplot(111)
    ax.plot(gdata.transpose())
    plt.title('{}\n({} CMC curves with {} experiments / curve)'.format(title, CMC_CNT, EXPERIMENTS))
    ax.legend(('-stddev','avg','+stddev'), bbox_to_anchor=(1, -0.05), fancybox=True, shadow=True, ncol=5)
    plt.savefig(output_file)


def generate_curves(args):
    pkl_file = args

    with open(pkl_file, 'rb') as pkl_hdl:
        params = pickle.load(pkl_hdl)

    print(params['title'])

    # Require output dir
    out_dir = os.path.dirname(params['output_file'])
    if '/home/' in out_dir:
        out_dir = out_dir.replace('/home/', '/Users/')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Plot experiment results
    figure = plt.figure()
    ax = plt.subplot(111)
    ax.plot(params['gdata'].transpose())
    plt.title('{title}\n({CMC_CNT} CMC curves with {EXPERIMENTS} experiments / curve)'.format(**params))
    ax.legend(('-stddev', 'avg', '+stddev'), bbox_to_anchor=(1, -0.05), fancybox=True, shadow=True, ncol=5)
    plt.savefig(params['output_file'].replace('/home/', '/Users/'))

# In[ ]:

# Run all the experiments

def get_jobs(index):
    for training_set in index:
        for test_set in index[training_set]:
            feature_file = index[training_set][test_set]
            title = "Processing {} using {} trained network".format(test_set, training_set)
            out_file = "{}/{}_{}".format(OUT_DIR, test_set, training_set)
            yield feature_file, title, out_file

def get_results(results_dir):
    for pkl in glob.glob(os.path.join(results_dir, '*.pkl')):
        yield pkl

pool = multiprocessing.Pool(4)
try:
    for p in get_results(OUT_DIR):
        generate_curves(p)
    # map(generate_curves, get_results(OUT_DIR))
finally:
    pool.close()


# In[ ]:



