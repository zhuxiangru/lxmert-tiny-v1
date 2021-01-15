# The name of experiment
name=lxmert

# Create dirs and make backup
output=snap/pretrain/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# Pre-training
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/pretrain/lxmert_pretrain.py \
    --taskMaskLM --taskObjPredict --taskMatched --taskQA \
    --visualLosses obj,attr,feat \
    --wordMaskRate 0.15 --objMaskRate 0.15 \
    --train mscoco_train,mscoco_nominival,vgnococo --valid mscoco_minival \
    --llayers 4 --xlayers 2 --rlayers 2 \
    --fromScratch \
    --batchSize 64 --optim bert --lr 1e-4 --epochs 20 \
    --tqdm --output $output ${@:2}

