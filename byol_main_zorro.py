
from data_utils.multi_view_data_injector import MultiViewDataInjector
from data_utils.transforms import get_data_transforms, get_data_transforms_zorro, get_data_transforms_zorro_eval, get_data_transforms_hulk, get_data_transforms_eval_hulk, get_data_transforms_eval_robin, get_data_transforms_hulk_without_normalization
#from data_utils.usecase1 import UseCase1
from data_utils.hulk import Hulk
from data_utils.zorro2 import Zorro
from data_utils.robin import Robin
#from data_utils.balanced_split import balanced_split, quotas_balanced_split
#from data_utils.data_setup_with_labels import create_dataloaders
from model_utils.mlp_head import MLPHead
from model_utils.resnet_base_network import ResNet
from model_utils.linear_classifier import LinearClassifier
#from engine_utils.byol_engine_no_class import BYOLTrainer
from engine_utils.byol_engine_single_class import BYOLTrainer
from astropy.utils.exceptions import AstropyWarning
import warnings
import os
import torch
import yaml
import argparse
#from astropy.io import fits

#TORUN: python byol/byol_main.py -f hulk-test-debug -e hulk-3-robin --epochs 1 three_smash.yaml

#TODO Add num_classes to HULK
#TODO Fix embedding generate even if possible image rendering > len(dataset)


warnings.simplefilter('ignore', category=AstropyWarning)

print(torch.__version__)
torch.manual_seed(0)

parser = argparse.ArgumentParser(
                    prog = 'Byol Trainer',
                    description = 'Train Byol',
                    epilog = 'epilogo')

# example command line

parser.add_argument('current_config')           # yaml in the /configs folder
parser.add_argument('-e', '--experiment')       # exp name (invented, representative)
parser.add_argument('-f', '--folder')           # where to log results
parser.add_argument('-epcs', '--epochs', default=30)      # option that takes a value
parser.add_argument('-r', '--from_checkpoint', action='store_true')      # option that takes a value

args = parser.parse_args()

print(args.epochs)
data_dir = "."
src_dir = "byol"
config_dir = "configs"

#split_mode = "balanced"
split_portion = 0.9
split_mode = "fixed"

config_path = os.path.join( src_dir, config_dir, args.current_config )
transforms_path = os.path.join( src_dir, "data_utils", "transforms.py" ) # To be copied by summarywriter in logs
main_script_path = os.path.join( src_dir, "byol_main_zorro.py" ) # To be copied by summarywriter in logs

#network = "resnet18"

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")
    
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    #data_transform = get_data_transforms_hulk(**config['network']['input_shape'])
    data_transform = get_data_transforms_zorro(**config['network']['input_shape'])
    data_transform_eval = get_data_transforms_zorro_eval(**config['network']['input_shape'])

    data_path   = os.path.join(data_dir, config["dataset"])
    #eval_data_path   = os.path.join(data_dir, "2-ROBIN")
    eval_data_path   = os.path.join(data_dir, config["dataset"])

    #data_custom = Hulk(targ_dir = data_path, transform = MultiViewDataInjector([data_transform, data_transform]))

    """
    train_dataset = Zorro('1-ZORRO/data', '1-ZORRO/data/train_all.txt', 
            MultiViewDataInjector([data_transform, data_transform]), config["network"]["input_shape"]["width"]
        )
    validation_dataset = Zorro('1-ZORRO/data', '1-ZORRO/data/val_all.txt',
            MultiViewDataInjector([data_transform, data_transform]), config["network"]["input_shape"]["width"]
        )
    """
    data_path = "1-ZORRO/data"
    train_dataset = Zorro(targ_dir = data_path, transform = MultiViewDataInjector([data_transform, data_transform]),
                          datalist="train.json"
        )
    validation_dataset = Zorro(targ_dir = data_path, transform = MultiViewDataInjector([data_transform, data_transform]),
                          datalist="validation.json"
        )
    
    eval_dataset = Zorro(targ_dir = data_path, transform = data_transform_eval,
                          datalist="validation.json"
        )
    #train_dataset = torch.utils.data.Subset(train_dataset, range(1,10000))

    
    #eval_data = Hulk(targ_dir = data_path, transform = data_transform_eval, datalist="labeled.json")
    #eval_data = torch.utils.data.Subset(eval_data, range(1,23000))
    #eval_data = Robin(targ_dir = eval_data_path, transform = data_transform_eval)
    
    #train_dataset, test_dataset = torch.utils.data.random_split(data_sampled, [train_split, test_split])
    
    #data_sampled = data_custom
    #eval_data_sampled = eval_data
    
    #eval_data_sampled = torch.utils.data.Subset(eval_data, range(1,11000))
    #train_split = int(0.9 * len(data_sampled))
    #test_split  = len(data_sampled) - train_split
    #train_dataset, test_dataset = torch.utils.data.random_split(data_sampled, [train_split, test_split])

    print(len(train_dataset))
    print(len(validation_dataset))
    #print(len(eval_data))
    
    # Twin networks
    online_network = ResNet(**config['network']).to(device)
    target_network = ResNet(**config['network']).to(device)

    predictor = MLPHead(in_channels=online_network.projetion.net[-1].out_features,
                        **config['network']['projection_head']).to(device)
    optimizer = torch.optim.SGD(list(online_network.parameters()) + list(predictor.parameters()),
                                **config['optimizer']['params'])
    
    linear_classificator = torch.nn.Linear(
        online_network.repr_shape, eval_dataset.num_classes).to(device)
    
    optimizer_classificator = torch.optim.SGD(list(linear_classificator.parameters()),
                                **config['optimizer']['params'])
    config["trainer"]["max_epochs"] = int(args.epochs)
        
    if args.from_checkpoint:
        recover_from_checkpoint = True
    else:
        recover_from_checkpoint = False

    trainer = BYOLTrainer(online_network=online_network,
                          target_network=target_network,
                          linear_classifier=linear_classificator,
                          multilabel_linear_classificator = None,
                          optimizer=optimizer,
                          optimizer_classificator=optimizer_classificator,
                          predictor=predictor,
                          device=device,
                          logs_folder=args.folder,
                          exp_name=args.experiment,
                          config_path=config_path,
                          transforms_path=transforms_path,
                          main_script_path = main_script_path,
                          pretrained=config["network"]["pretrained_weights"],
                          img_size = config["network"]["input_shape"]["width"],
                          recover_from_checkpoint = recover_from_checkpoint,
                          preview_shape = config["network"]["preview_shape"],
                          classes = None,
                          **config["trainer"])

    trainer.train(train_dataset, validation_dataset, eval_dataset)
    

if __name__ == '__main__':
    main()