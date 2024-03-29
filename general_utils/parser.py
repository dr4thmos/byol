import argparse

def parse():
    parser = argparse.ArgumentParser(
                    prog = 'Byol Trainer',
                    description = 'Train Byol',
                    epilog = 'epilogo')
                    
    parser.add_argument('current_config')           # yaml in the /configs folder
    parser.add_argument('-e', '--experiment')       # exp name (invented, representative)
    parser.add_argument('-f', '--folder')           # where to log results
    parser.add_argument('-epcs', '--epochs', default=30)      # option that takes a value
    parser.add_argument('-t', '--test_every_n_epochs', default=5)      # option that takes a value
    parser.add_argument('-r', '--from_checkpoint', action='store_true')      # option that takes a value
    args = parser.parse_args()
    return args