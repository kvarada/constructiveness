import argparse, sys
from constructiveness_data_collector import *
sys.path.append('../../')
from config import Config

def get_arguments():
    parser = argparse.ArgumentParser(description='Create training and test datafiles for constructiveness')
    
    parser.add_argument('--train_csv', '-tr', type=str, dest='train_csv', action='store',                     
                        default = Config.TRAIN_PATH + 'SOCC_NYT_picks_constructive_YNACC_non_constructive.csv',
                        help="The training data output CSV containing instances for constructive and non-constructive comments.")

    parser.add_argument('--test_csv', '-te', type=str, dest='test_csv', action='store',
                        default = Config.TEST_PATH + 'SOCC_constructiveness.csv',
                        help="The test data output CSV containing instances for constructive and non-constructive comments.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()
    cdc = ConstructivenessDataCollector()

    cdc.collect_training_data_from_CSV(Config.SOCC_ANNOTATED_CONSTRUCTIVENESS_12000, source = 'SOCC')
    
    # Collect training data
    positive_examples_csvs = [[Config.NYT_PICKS_SFU, 1.0], [Config.NYT_PICKS_COMMENTIQ, 1.0]]
    negative_examples_csvs = [[Config.YNACC_EXPERT_ANNOTATIONS, 1.0], [Config.YNACC_MTURK_ANNOTATIONS, 0.43] ]
    cdc.collect_train_data(positive_examples_csvs, negative_examples_csvs)

    cdc.write_csv(args.train_csv)

    # Collect test data
    #test_csvs = [[args.SOCC_annotated_csv,1.0]]
    #cdc.collect_test_data(test_csvs, args.test_csv)
