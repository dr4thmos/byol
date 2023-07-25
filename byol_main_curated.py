
from data_utils.multi_view_data_injector import MultiViewDataInjector
from data_utils.transforms import get_data_transforms, get_data_transforms_hulk, get_data_transforms_eval_hulk, get_data_transforms_eval_robin, get_data_transforms_hulk_without_normalization
#from data_utils.usecase1 import UseCase1
from data_utils.hulk import Hulk
from data_utils.robin import Robin
#from data_utils.balanced_split import balanced_split, quotas_balanced_split
#from data_utils.data_setup_with_labels import create_dataloaders
from model_utils.mlp_head import MLPHead
from model_utils.resnet_base_network import ResNet
from model_utils.linear_classifier import LinearClassifier
from engine_utils.byol_engine import BYOLTrainer
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
main_script_path = os.path.join( src_dir, "byol_main.py" ) # To be copied by summarywriter in logs

#network = "resnet18"

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")
    
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    data_transform = get_data_transforms_hulk(**config['network']['input_shape'])
    #data_transform = get_data_transforms_hulk_without_normalization(**config['network']['input_shape'])
    data_transform_eval = get_data_transforms_eval_hulk(**config['network'])

    data_path   = os.path.join(data_dir, config["dataset"])
    #eval_data_path   = os.path.join(data_dir, "2-ROBIN")
    eval_data_path   = os.path.join(data_dir, config["dataset"])

    #data_custom = Hulk(targ_dir = data_path, transform = MultiViewDataInjector([data_transform, data_transform]))
    validation_dataset = Hulk(targ_dir = data_path,
                              transform = MultiViewDataInjector([data_transform, data_transform]),
                              datalist="labeled.json")
    
    train_dataset = Hulk(targ_dir = data_path,
                         transform = MultiViewDataInjector([data_transform, data_transform]),
                         datalist="unlabeled.json")
    #train_dataset = torch.utils.data.Subset(train_dataset, range(1,200000))

    
    eval_data = Hulk(targ_dir = data_path, transform = data_transform_eval, datalist="labeled.json")
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
    print(len(eval_data))
    
    # online network
    online_network = ResNet(**config['network']).to(device)
    target_network = ResNet(**config['network']).to(device)
    predictor = MLPHead(in_channels=online_network.projetion.net[-1].out_features,
                        **config['network']['projection_head']).to(device)
    optimizer = torch.optim.SGD(list(online_network.parameters()) + list(predictor.parameters()),
                                **config['optimizer']['params'])
    
    linear_classificator = torch.nn.Linear(
        online_network.repr_shape, eval_data.num_classes).to(device)
    multilabel_linear_classificator = LinearClassifier(online_network.repr_shape, eval_data.num_classes).to(device)
    optimizer_classificator = torch.optim.SGD(list(multilabel_linear_classificator.parameters()),
                                **config['optimizer']['params'])
    
    config["trainer"]["max_epochs"] = int(args.epochs)
        
    if args.from_checkpoint:
        recover_from_checkpoint = True
    else:
        recover_from_checkpoint = False

    trainer = BYOLTrainer(online_network=online_network,
                          target_network=target_network,
                          linear_classifier=linear_classificator,
                          multilabel_linear_classificator = multilabel_linear_classificator,
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
                          classes = validation_dataset.classes,
                          optimizer_params = config['optimizer']['params'],
                          **config["trainer"])

    trainer.train(train_dataset, validation_dataset, eval_data)
    
    """
    pretrained_folder = config['network']['fine_tune_from']

    # load pre-trained model if defined
    if pretrained_folder:
        try:
            checkpoints_folder = os.path.join('./runs', pretrained_folder, 'checkpoints')

            # load pre-trained parameters
            load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'model.pth')),
                                     map_location=torch.device(torch.device(device)))

            online_network.load_state_dict(load_params['online_network_state_dict'])

        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")
    """

if __name__ == '__main__':
    main()