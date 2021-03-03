# lxmert-tiny-v1

在lxmert原版模型上修改以及测评。
原版：https://github.com/airsplay/lxmert

## one-step demo：没有抽取出obj feature && bbox + 用预训练任务的图文匹配参数task_matched

bash run/image_text_matching_test_no_obj_demo.bash 0 image_text_matching_lxr955_results  --test matching_test


## two-step large-scale test: 没有没有抽取出obj feature && bbox + 用预训练任务的图文匹配参数task_matched

###step1:
如果机器没安装过docker镜像，则生成matching_val_no_obj_obj36.tsv
sudo docker pull airsplay/bottom-up-attention

进入docker
docker run --gpus all -v /path/to/nlvr2/images:/workspace/images:ro -v /path/to/lxrt_public/data/nlvr2_imgfeat:/workspace/features --rm -it airsplay/bottom-up-attention bash
docker run --gpus all -v /mnt/data/zxr/work/lxmert/data/image_text_matching_no_obj_images:/workspace/images:ro -v /mnt/data/zxr/work/lxmert/data/image_text_matching_no_obj_imgfeat:/workspace/features --rm -it airsplay/bottom-up-attention bash


cd /workspace/features
CUDA_VISIBLE_DEVICES=0 python extract_demo_image.py --split small 

###step2：
cd data/image_text_matching_no_obj_images/img_feat/
ln -s ../../image_text_matching_no_obj_imgfeat/small_obj36.tsv small_obj36.tsv
测试一下:（--with_feat表示用已经抽取好的feature，而不是再跑frcnn即时抽取）
bash run/image_text_matching_test_no_obj_demo.bash 0 image_text_matching_lxr955_results  --test matching_test_demo_v2 --with_feat
运行：(缩略图，且用图像相似度，from sunpenglei)
bash run/image_text_matching_test_no_obj_demo.bash 0 image_text_matching_lxr955_results  --test matching_val --with_feat
运行：(缩略图，且用图像聚类，from zhuxiangru)


## 已经抽取出obj feature && bbox + 用预训练任务的图文匹配参数task_matched

### test:

cd lxmert
bash run/image_text_matching_test.bash 0 image_text_matching_lxr955_results  --test matching_test

### evaluation: 

cd lxmert
bash run/image_text_matching_test.bash 0 image_text_matching_lxr955_results  --test matching_val

## 已经抽取出obj feature && bbox + 用训练完成的模型（没有预训练任务层参数）的lang表示和vision表示做cos相似度

### test: （用预训练后的LXRTModel,不包含四个预训练任务的参数）
cd lxmert
bash run/image_text_matching_test_cos_base.bash 0 image_text_matching_lxr955_results  --test matching_test

cd lxmert
bash run/image_text_matching_test_cos_vqa.bash 0 image_text_matching_lxr955_results  --test matching_test

或者
### test: （用预训练后的LXRTModel,不包含四个预训练任务的参数）base model
cd lxmert-huggingface
python demo/demo_test_base.py 

### test: （用预训练后的LXRTModel,不包含四个预训练任务的参数）vqa-finetuned model
cd lxmert-huggingface
python demo/demo_test_vqa.py 


## 注意事项

### 1. frcnn较慢。待评估的图片提前抽取出特征放置一文件。

注意，这里如果用test, 则tsv文件会读取整个test2015_obj36.tsv，太大了，时间较长

'''
bash run/image_text_matching_test.bash 0 image_text_matching_lxr422_results  --test test    
'''

36 boxes
https://uc84c267efcd4f9c620ecdab7058.dl.dropboxusercontent.com/cd/0/get/BHWuLqmvZPfNcNLhOEL4Z0_NgR7DOQdEjKieE60nOLNgIiIdQt-ff8RFKTfqyfCy9euyFwr3uVAazPVBpDmXMoD6vFHnu1xAgkuuKZf7zeGxxOa3ih_ETV2rV3uIb_zfE2k/file?dl=1#

10-100 boxes
https://uc7eb2391c127b87500b8bbaa544.dl.dropboxusercontent.com/cd/0/get/BHXzAol5GhWmrzX_sIh5sdVws6ZLwcWQmGk6bjdEf5Z1YqVxn719aWxa_WswokOeSTrgsicNgKp898Hjr3qFc6Yf3Ke3FEYZ2CP1wtdouzF5EntEfxeOjGpzrsH0TdN2LcA/file?dl=1#

