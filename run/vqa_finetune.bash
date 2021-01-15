# The name of this experiment.
name=$2

# Save logs and models under snap/vqa; make backup.
output=snap/vqa/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/vqa.py \
    --train train,nominival --valid minival  \
    --llayers 4 --xlayers 2 --rlayers 2 \
    --loadLXMERTQA snap/pretrained/model \
    --batchSize 32 --optim bert --lr 5e-5 --epochs 4 \
    --tqdm --output $output ${@:3}
