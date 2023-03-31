#!/bin/sh
#python dynamic_input_self/byol_main.py 1channel_histeq.yaml
#python dynamic_input_self/byol_main.py 1channel_minmax.yaml
#python dynamic_input_self/byol_main.py 1channel_sigma3.yaml
#python dynamic_input_self/byol_main.py 3_channel.yaml

python dynamic_input_self/byol_main.py 1channel_histeq.yaml -f right_runs --epochs 125
python dynamic_input_self/byol_main.py 1channel_minmax.yaml -f right_runs --epochs 125
python dynamic_input_self/byol_main.py 1channel_sigma3.yaml -f right_runs --epochs 125
python dynamic_input_self/byol_main.py 3_channel.yaml -f right_runs --epochs 125
python dynamic_input_self/byol_main.py 3_channel_fine_tune.yaml -f right_runs --epochs 125


#python dynamic_input_self/byol_main.py 3_channel_fine_tune.yaml -f test_multiple_labels --epochs 10