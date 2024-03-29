# The name of this experiment.
name=$2

# Save logs and models under snap/vqa; make backup.
output=snap/image_text_matching/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/eval_matching_obj/image_text_matching_base_vqa.py \
    --tiny --train train --valid ""  \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --loadLXMERT snap/unc_pretrained_base/model \
    --batchSize 32 --optim bert --lr 5e-5 --epochs 4 \
    --tqdm --output $output ${@:3}
