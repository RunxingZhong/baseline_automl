import argparse
from lib.util import timeit
from lib.automl import AutoML
import datetime
from lib.util import timeit, log, Config

@timeit
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['classification', 'regression'])
    parser.add_argument('--model-dir')
    parser.add_argument('--train-csv')
    parser.add_argument('--test-csv')
    parser.add_argument('--prediction-csv')
    args = parser.parse_args()

    automl = AutoML(args.model_dir)

    log('####### cur time = ' + str(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")))
    #
    automl.config['path_pred'] = args.model_dir
    
    if args.train_csv is not None:
        log('automl train...')
        automl.train(args.train_csv, args.mode)
        automl.save()
    elif args.test_csv is not None:
        log('automl predict...')
        automl.load()
        automl.predict(args.test_csv, args.prediction_csv)
    else:
        exit(1)
        
    log('####### cur time = ' + str(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")))


if __name__ == '__main__':
    main()
