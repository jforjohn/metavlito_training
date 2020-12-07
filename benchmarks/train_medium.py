import argparse
import datetime
import sys
import os

PROJECT_PATH = os.path.abspath(os.path.join(__file__, *(os.path.pardir for _ in range(2))))
sys.path.append(PROJECT_PATH)

if sys.version_info < (3, 0):
    print("Sorry, requires Python 3.x, not Python 2.x\n")
    sys.exit(1)

from consts.config_loader import load
from benchmarks.src.training import classification_train
from consts.paths import DatasetFactory

def constructDatasetPaths(config_data, no_buckets):
    datasets_dir = config_data.get('datasets_dir')
    dataset_name = config_data.get('dataset_name')

    datasets_path = os.path.join(PROJECT_PATH,
                            datasets_dir, dataset_name
                            )
    
    csv_file = os.path.join(datasets_path,
                            dataset_name + '.csv')
    
    images_path = os.path.join(datasets_path,
                               'data')

    if no_buckets:
        buckets_path = os.path.join(datasets_path,
                                   'buckets')
        buckets_train = os.path.join(buckets_path,
                                     'buckets_train_%s.npy' %(no_buckets))
        buckets_val = os.path.join(buckets_path,
                                   'buckets_val_%s.npy' %(no_buckets))
        b_extras = os.path.join(buckets_path,
                                'b_extras_%s.npy' %(no_buckets))
        buckets_files = {
            'train': buckets_train,
            'validation': buckets_val,
            #'b_extras': b_extras
        }
    else:
        buckets_files = {}


    return csv_file, images_path, buckets_files

if __name__ == "__main__":
    print("Project path: '{}'".format(PROJECT_PATH))

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="specify the location of the  config file", type=str, default="consts/config.cfg"
        )
    args, _ = parser.parse_known_args()

    config_file = args.config
    config = load(config_file)

    #medium_ds = DatasetFactory(DatasetFactory.METH_MEDIUM)

    retrain = config.get('options', 'retrain')
    no_buckets = config.getint('options', 'no_buckets')
    #retrain_buckets = config.getboolean('options', 'retrain_buckets')

    csv_file, images_path, buckets_files = constructDatasetPaths(config['data'], no_buckets)
    
    kwargs = {
        'retrain': retrain,
        'buckets_files': buckets_files
    }
    
    experiments_dir = config.get('data', 'experiments_dir')
    experiments_path = os.path.join(PROJECT_PATH, experiments_dir)
    try:
        # Create experiments directory
        os.mkdir(experiments_path)
    except FileExistsError:
        pass
        #print("Directory " , experiments_path ,  " already exists")
    
    current_date = '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now())
    experiment_filename = config.get('data', 'experiment_name')
    # if experiment filename not defined
    # generate name based on the current date
    if not experiment_filename:
        experiment_filename = 'medium_%s' %(current_date)

    experiment_path = os.path.join(experiments_path, experiment_filename)
    # Create single experiment directory
    try:
        # Create experiments directory
        os.mkdir(experiment_path)
        print('Experiment path:', experiment_path)
    except FileExistsError:
        print("Directory " , experiment_path ,  " already exists")

    model_path = os.path.join(experiment_path, 'model.ckpt')
    summaries_path = os.path.join(experiment_path, 'sum_{}'.format(experiment_filename))
    
    print("Options:")
    print("retrain: '{}'".format(retrain))
    print("buckets_files '{}'".format(buckets_files))
    if retrain:
        classification_train(csv_file, images_path, retrain, summaries_path, config['training'], **kwargs)
    else:
        classification_train(csv_file, images_path, model_path, summaries_path, config['training'], **kwargs)
