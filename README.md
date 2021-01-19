# lxmert-tiny-v1

# 已经抽取出obj feature && bbox + 用预训练任务的图文匹配参数task_matched

## test:

cd lxmert
bash run/image_text_matching_test.bash 0 image_text_matching_lxr955_results  --test matching_test

## evaluation: 

cd lxmert
bash run/image_text_matching_test.bash 0 image_text_matching_lxr955_results  --test matching_val

# 已经抽取出obj feature && bbox + 用训练完成的模型（没有预训练任务层参数）的lang表示和vision表示做cos相似度

## test: （用预训练后的LXRTModel,不包含四个预训练任务的参数）
cd lxmert
bash run/image_text_matching_test_cos.bash 0 image_text_matching_lxr955_results  --test matching_test

或者
## test: （用预训练后的LXRTModel,不包含四个预训练任务的参数）base model
cd lxmert-huggingface
python demo/demo_test_base.py 

## test: （用预训练后的LXRTModel,不包含四个预训练任务的参数）vqa-finetuned model
cd lxmert-huggingface
python demo/demo_test_vqa.py 


# 注意事项

## 1. frcnn较慢。待评估的图片提前抽取出特征放置一文件。

注意，这里如果用test, 则tsv文件会读取整个test2015_obj36.tsv，太大了，时间较长

'''
bash run/image_text_matching_test.bash 0 image_text_matching_lxr422_results  --test test    
'''
