
from data_utils.transforms import get_data_transforms_curated, get_data_transforms_eval_curated
from data_utils.multi_view_data_injector import MultiViewDataInjector
from data_utils.curated import Curated
from model_utils.mlp_head import MLPHead
from model_utils.resnet_base_network import ResNet
from engine_utils.byol_engine import BYOLTrainer
from astropy.utils.exceptions import AstropyWarning
import warnings
warnings.simplefilter('ignore', category=AstropyWarning)
import os
import torch
import yaml
import argparse
from astropy.io import fits

#TORUN: python byol/byol_main_curated.py -f curated -e test1 -epcs 3 curatino.yaml

print(torch.__version__)
torch.manual_seed(0)

parser = argparse.ArgumentParser(
                    prog = 'Byol Trainer',
                    description = 'Train Byol',
                    epilog = 'epilogo')

parser.add_argument('current_config')           # yaml in the /configs folder
parser.add_argument('-e', '--experiment')       # exp name (invented, representative)
parser.add_argument('-f', '--folder')           # where to log results
parser.add_argument('-epcs', '--epochs', default=30)      # option that takes a value
parser.add_argument('-r', '--from_checkpoint', action='store_true')      # option that takes a value

args = parser.parse_args()

data_dir = "."
src_dir = "byol"
config_dir = os.path.join( src_dir, "configs")

config_path = os.path.join( config_dir, args.current_config )
transforms_path = os.path.join( src_dir, "data_utils", "transforms.py" ) # To be copied by summarywriter in logs
main_script_path = os.path.join( src_dir, "byol_main.py" ) # To be copied by summarywriter in logs

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")
    print(f"Number of epochs:{args.epochs}")
    
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    data_transform      = get_data_transforms_curated(**config['network']['input_shape'])
    data_transform_eval = get_data_transforms_eval_curated(**config['network'])

    data_path   = os.path.join(data_dir, config["dataset"])

    train_dataset       = Curated(targ_dir = data_path,
                                transform = MultiViewDataInjector([data_transform, data_transform]))

    validation_dataset  = Curated(targ_dir = data_path,
                                transform = MultiViewDataInjector([data_transform, data_transform]))
    
    #train_dataset = torch.utils.data.Subset(train_dataset, range(1,200000))

    print(len(train_dataset))
    print(len(validation_dataset))
    
    # online network
    online_network = ResNet(**config['network']).to(device)
    target_network = ResNet(**config['network']).to(device)

    predictor = MLPHead(in_channels=online_network.projetion.net[-1].out_features,
                        **config['network']['projection_head']).to(device)
    optimizer = torch.optim.SGD(list(online_network.parameters()) + list(predictor.parameters()),
                                **config['optimizer']['params'])
    
    config["trainer"]["max_epochs"] = int(args.epochs)
        
    if args.from_checkpoint:
        recover_from_checkpoint = True
    else:
        recover_from_checkpoint = False

    trainer = BYOLTrainer(online_network=online_network,
                          target_network=target_network,
                          linear_classifier=None,
                          multilabel_linear_classificator = None,
                          optimizer=optimizer,
                          optimizer_classificator=None,
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
                          classes = validation_dataset.classes,
                          optimizer_params = config['optimizer']['params'],
                          **config["trainer"])

    trainer.train(train_dataset, validation_dataset, None)

if __name__ == '__main__':
    main()