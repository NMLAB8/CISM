#!/bin/bash

echo "Starting inference..."


echo "Running inference for Dizi"
python -m scnet.inference \
    --input_dir test_music/dizi \
    --output_dir test_music_output/dizi \
    --checkpoint_path checkpoint/dizi_checkpoint.th \
    --config_path conf/conf_test/0.1750,0.3920,0.4330.yaml


echo "Running inference for Guzheng"
python -m scnet.inference \
    --input_dir test_music/guzheng \
    --output_dir test_music_output/guzheng \
    --checkpoint_path checkpoint/guzheng_checkpoint.th \
    --config_path conf/conf_test/0.1750,0.3920,0.4330.yaml

echo "Running inference for Pipa"
python -m scnet.inference \
    --input_dir test_music/pipa \
    --output_dir test_music_output/pipa \
    --checkpoint_path checkpoint/pipa_checkpoint.th \
    --config_path conf/conf_test/0.1750,0.3920,0.4330.yaml


echo "Running inference for Xiao"
python -m scnet.inference \
    --input_dir test_music/xiao \
    --output_dir test_music_output/xiao \
    --checkpoint_path checkpoint/xiao_checkpoint.th \
    --config_path conf/conf_test/0.1750,0.3920,0.4330.yaml

echo "All inferences completed!"
