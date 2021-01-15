# The name of this experiment.
name=$2

# Save logs and models under snap/gqa; make backup.
output=snap/gqa/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/gqa.py \
    --train train,valid --valid testdev \
    --llayers 4 --xlayers 2 --rlayers 2 \
    --loadLXMERTQA snap/pretrained/model \
    --batchSize 32 --optim bert --lr 1e-5 --epochs 4 \
    --tqdm --output $output ${@:3}
