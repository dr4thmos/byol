
from data_utils.multi_view_data_injector import MultiViewDataInjector
from data_utils.transforms import get_data_transforms
from data_utils.usecase1 import UseCase1
from data_utils.balanced_split import balanced_split
#from data_utils.data_setup_with_labels import create_dataloaders
from model_utils.mlp_head import MLPHead
from model_utils.resnet_base_network import ResNet
from engine_utils.byol_engine import BYOLTrainer
#from torchvision.transforms import transforms

import os
import torch
import yaml
import argparse

#sys.path.append('../')

print(torch.__version__)
torch.manual_seed(0)

parser = argparse.ArgumentParser(
                    prog = 'Byol Trainer',
                    description = 'Train Byol',
                    epilog = 'epilogo')

parser.add_argument('current_config')           # positional argument
#parser.add_argument('-c', '--count')      # option that takes a value
#parser.add_argument('-v', '--verbose',
#                    action='store_true')  # on/off flag
args = parser.parse_args()
data_dir = "data"
src_dir = "dynamic_input_self"
config_dir = "configs"

#current_config = "exp1" + ".yaml"
#current_data = "3_channel_croppedbackyard"
split_mode = "balanced"
split_portion = 0.8

config_path = os.path.join(src_dir, config_dir, args.current_config)

network = "resnet18"

def main():
    config_name = "template"

    custom_config = """
        network: resnet18
        fine_tune_from: None
        dataset: 3_channel_croppedbackyard
        split_mode: balanced
        split_portion: 0.8
        projection_head: 
            mlp_hidden_size: 512
            projection_size: 128
    """

    transforms_config = """
        rotate:
            degrees: 90.0
            interpolation: BILINEAR
            expand: True
        conditioned_resize: True
        shift_and_pad_to_size: True
        resize: (96,96,3)
        flip:
            horizontal: True
            vertical: True
    """

    trainer_config = """
        trainer:
            batch_size: 128
            m: 0.996 # momentum update
            checkpoint_interval: 500000
            max_epochs: 1500
            num_workers: 8

        optimizer:
            params:
                lr: 0.03
                momentum: 0.9
                weight_decay: 0.0004
    """

    output_config = """
        log:
            loss: True
            accuracy: False

        embeddings: True
        
        evaluation:
            mlp_finetune: True
    """

    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")

    data_transform = get_data_transforms(**config['network'])

    #data_path = "data/3_channel_croppedbackyard"

    data_path = os.path.join(data_dir, config["dataset"])
    data_custom = UseCase1(targ_dir = data_path, transform = MultiViewDataInjector([data_transform, data_transform]))

    #split = 0.8

    train_dataset, test_dataset = balanced_split(data_custom, split_portion)
    # online network
    online_network = ResNet(**config['network']).to(device)
    # Eventually load weights
    target_network = ResNet(**config['network']).to(device)
    predictor = MLPHead(in_channels=online_network.projetion.net[-1].out_features,
                        **config['network']['projection_head']).to(device)
    optimizer = torch.optim.SGD(list(online_network.parameters()) + list(predictor.parameters()),
                                **config['optimizer']['params'])
    trainer = BYOLTrainer(online_network=online_network,
                          target_network=target_network,
                          optimizer=optimizer,
                          predictor=predictor,
                          device=device,
                          exp_name=config["dataset"],
                          config_path=config_path,
                          **config["trainer"])

    trainer.train(train_dataset, test_dataset)
    
    

if __name__ == '__main__':
    main()