#!/bin/sh
#python dynamic_input_self/eval_byol.py -f benchmark -e 1_channel_histeq
#python dynamic_input_self/eval_byol.py -f benchmark -e 1_channel_minmax
#python dynamic_input_self/eval_byol.py -f benchmark -e 1_channel_sigma3
#python dynamic_input_self/eval_byol.py -f benchmark -e 1_channel_sigma3_minmax_histeq
#python dynamic_input_self/eval_byol.py -f benchmark -e 3_channel_minmax_sigma3_histeq
#python dynamic_input_self/eval_byol.py -f benchmark -e 3_channel_minmax_sigma3_histeq_finetune

python dynamic_input_self/eval_byol.py -f test_run -e 1_channel_histeq --epochs 30
python dynamic_input_self/eval_byol.py -f test_run -e 1_channel_minmax --epochs 30
python dynamic_input_self/eval_byol.py -f test_run -e 1_channel_sigma3 --epochs 30
python dynamic_input_self/eval_byol.py -f test_run -e 1_channel_sigma3_minmax_histeq --epochs 30
python dynamic_input_self/eval_byol.py -f test_run -e 3_channel_minmax_sigma3_histeq --epochs 30