# The name of this experiment.
name=$2

# Save logs and models under snap/nlvr2; make backup.
output=snap/nlvr2/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/nlvr2.py \
    --tiny --llayers 4 --xlayers 2 --rlayers 2 \
    --tqdm --output $output ${@:3}
