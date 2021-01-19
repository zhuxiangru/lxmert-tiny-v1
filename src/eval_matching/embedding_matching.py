# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 17:36:19 2020

@author: zhuxiangru
"""
from transformers import BertModel, BertTokenizer, BertConfig, RobertaTokenizer, RobertaModel
import torch
import logging
import math
from collections import defaultdict

# 采用AutoTokenizer、AutoModelForMaskedLM下载的
# from transformers import AutoTokenizer, AutoModelForMaskedLM
# tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
# tokenizer.save_pretrained("/mnt/g/program/text_entailment/model/roberta-large/tokenizer")  # 保存到指定目录
# model = AutoModelForMaskedLM.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
# model.save_pretrained("/mnt/g/program/text_entailment/model/roberta-large/") # 保存到指定目录

# download：bert-base-chinese
# tokenizer = BertTokenizer.from_pretrained('./model/bert/tokenizer')
# model = BertModel.from_pretrained('./model/bert', return_dict=True)

# download：hfl/chinese-roberta-wwm-ext
# https://github.com/ymcui/Chinese-BERT-wwm#%E5%BF%AB%E9%80%9F%E5%8A%A0%E8%BD%BD
# tokenizer = BertTokenizer.from_pretrained('./model/roberta/tokenizer')
# model = BertModel.from_pretrained('./model/roberta', return_dict=True)

# download：hfl/chinese-roberta-wwm-ext-large
# https://github.com/ymcui/Chinese-BERT-wwm#%E5%BF%AB%E9%80%9F%E5%8A%A0%E8%BD%BD
# tokenizer = BertTokenizer.from_pretrained('./model/roberta-large/tokenizer')
# model = BertModel.from_pretrained('./model/roberta-large', return_dict=True)

class SentencePairMatch():
    def __init__(self, model_name="bert", language="en"):
        self.language = language
        if language == "zh":
            if model_name == "roberta-large":
                # download：hfl/chinese-roberta-wwm-ext-large
                # https://github.com/ymcui/Chinese-BERT-wwm#%E5%BF%AB%E9%80%9F%E5%8A%A0%E8%BD%BD
                self.tokenizer = BertTokenizer.from_pretrained('./model/tokenizer')
                self.model = BertModel.from_pretrained('./model/chinese-roberta-wwm-ext-large', return_dict=True)
            elif model_name == "roberta":
                # download：hfl/chinese-roberta-wwm-ext
                # https://github.com/ymcui/Chinese-BERT-wwm#%E5%BF%AB%E9%80%9F%E5%8A%A0%E8%BD%BD
                self.tokenizer = BertTokenizer.from_pretrained('./model/tokenizer')
                self.model = BertModel.from_pretrained('./model/chinese-roberta-wwm-ext', return_dict=True)
            elif model_name == "bert":
                # download：bert-base-chinese
                self.tokenizer = BertTokenizer.from_pretrained('./model/tokenizer')
                self.model = BertModel.from_pretrained('./model/chinese-bert-wwm-ext', return_dict=True)
            else:
                raise Exception("model type is not defined.")
        else:
            if model_name == "roberta-large":
                self.tokenizer = RobertaTokenizer.from_pretrained('./model/roberta-large')
                self.model = RobertaModel.from_pretrained('./model/roberta-large', return_dict=True)
            elif model_name == "roberta":
                self.tokenizer = RobertaTokenizer.from_pretrained('./model/roberta-base')
                self.model = RobertaModel.from_pretrained('./model/roberta-base', return_dict=True)
            elif model_name == "bert":
                #self.tokenizer = BertTokenizer.from_pretrained('./model/bert-base-uncased')
                self.tokenizer = BertTokenizer.from_pretrained('./model/bert-base-uncased/', do_lower_case=True)
                #self.model = BertModel.from_pretrained('./model/bert-base-uncased', return_dict=True)
                self.model = BertModel.from_pretrained('./model/bert-base-uncased/', return_dict=True)
            else:
                raise Exception("model type is not defined.")
        self.text_cache = defaultdict(list)

    def best_concept_match(self, sentence_list, concept_list):
        best_concept_match_set = defaultdict(dict)
        for sent in sentence_list:
            logging.debug("info=%s" % sent)

            max_score = -math.inf
            best_concept = ""
            for concept in concept_list:
                cos_sim = self.sentence_pair_match(sent, concept)
                logging.debug("concept=%s, similarity=%lf" % (concept, cos_sim))
                if cos_sim > max_score:
                    max_score = cos_sim
                    best_concept = concept
            logging.debug("info=%s, concept=%s" % (sent, best_concept))

            best_concept_match_set[best_concept][sent] = float(max_score)
        return best_concept_match_set

    def multi_sentence_match(self, sentence_a, sentence_b_list):
        max_score = -math.inf
        best_ans = ""
        for candidate_sent in sentence_b_list:
            cos_score = self.sentence_pair_match(sentence_a, candidate_sent)
            if cos_score > max_score:
                max_score = cos_score
                best_ans = candidate_sent
        return best_ans
    
    def multi_image_text_match(self, image_embedding, text_embedding_tensor, is_text_embedding = False):
        max_score = -math.inf
        best_ans = ""
        #for candidate_sent in text_embedding_list:
        print ("image_embedding.shape=", image_embedding.shape)
        if is_text_embedding == False:
            # text_embedding_tensor is a list of strings
            candidate_nums = len(text_embedding_tensor)
            candidate_text_embedding_list = [self.get_sentence_embedding(candidate_sent, search_cache = False) for candidate_sent in text_embedding_tensor]
            text_embedding_tensor = torch.cat(candidate_text_embedding_list, dim=0)
            text_embedding_tensor = text_embedding_tensor.repeat(image_embedding.shape[0], 1, 1)  # 有多少个问题就重复多少次，假设每个问题的选项都是一样的。实际情况是不一样的。
            print ("text_embedding_tensor.shape=", text_embedding_tensor.shape)
        else:
            candidate_nums = text_embedding_tensor.shape[0]
        
        image_embedding_tensor = image_embedding.unsqueeze(1)
        image_embedding_tensor = image_embedding_tensor.repeat(1,candidate_nums,1) # 重复选项次，和每个选项计算相似度，取最大
        print ("image_embedding_tensor=", image_embedding_tensor.shape)
        # sim = torch.cosine_similarity(image_embedding_tensor.cuda(), image_embedding_tensor.cuda(), dim=2)
        sim = torch.cosine_similarity(image_embedding_tensor.cuda(), text_embedding_tensor.cuda(), dim=2)
        print ("sim=", sim)
        cos_score = torch.max(sim, dim=1)
        print ("cos_score=", cos_score)
        return cos_score.values, cos_score.indices

    def sentence_pair_match(self, sentence_a, sentence_b, search_a_cache = False, search_b_cache = False):
        sentence_a_pooler_output = self.get_sentence_embedding(sentence_a, search_a_cache)
        sentence_b_pooler_output = self.get_sentence_embedding(sentence_b, search_b_cache)
        # get cos_similarity score
        cos_sim = torch.cosine_similarity(sentence_a_pooler_output, sentence_b_pooler_output, dim=1)
        return cos_sim
    


    def get_sentence_embedding(self, sentence, search_cache = False):
        if search_cache == True and sentence in self.text_cache:
            return self.text_cache[sentence]

        # get the [CLS] embedding of sentence(that is, pooler_output)
        if self.language == "zh":
            text_dict = self.tokenizer.encode_plus(sentence, add_special_tokens=True, return_attention_mask=True)
            input_ids = torch.tensor(text_dict['input_ids']).unsqueeze(0)
            token_type_ids = torch.tensor(text_dict['token_type_ids']).unsqueeze(0)
            attention_mask = torch.tensor(text_dict['attention_mask']).unsqueeze(0)
            # the format of res_a:(last_hidden_state, pooler_output, hidden_states, attentions)
            res = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            token_input = self.tokenizer(sentence, return_tensors='pt')
            res = self.model(**token_input)
        pooler_output = res.pooler_output
        if search_cache == True:
            self.text_cache[sentence] = pooler_output
        return pooler_output

    def get_text_cache(self):
        return self.text_cache


def test_fun_1():
    concepts = ["人物","演员","娱乐人物","歌手"]
    infos = ["1985年11月21日出生于中国香港，中国香港影视男演员、歌手、主持人，毕业于香港圣若瑟英文学校。",
             "2003年，因参加全球华人新秀香港区选拔赛而进入演艺圈。2006年成为Sun Boy’z组合一员。",
             "2008年开始独立发展，随后推出个人首张专辑《Will Power》，并获香港十大劲歌金曲颁奖礼最受欢迎男新人金奖及香港十大中文金曲最有前途新人金奖。",
             "2013年，陈伟霆将工作重心转移至中国内地，并凭借灾难电影《救火英雄》获得澳门国际电影节最佳男配角奖。",
             "2014年，因主演古装仙侠偶像剧《古剑奇谭》被内地观众熟知，并获得国剧盛典年度最受欢迎全能艺人及年度电视剧媒体最期待演员；同年，他还因励志喜剧电影《男人不可以穷》获得英国万像国际华语电影节最佳青年男演员奖。",
             "2015年，主演的民国偶像剧《活色生香》及古装仙侠剧《蜀山战纪之剑侠传奇》相继播出。",
             "2016年，凭借民国悬疑剧《老九门》获得颇高关注。2017年，他主演的古装奇幻权谋剧《醉玲珑》播出。",
             "2018年，陈伟霆主演的都市情感剧《南方有乔木》、魔幻电影《战神纪》及都市刑侦剧《橙红年代》相继播映。",
             "2020年8月27日，陈伟霆名列《2020福布斯中国名人榜》第40位。"]

    matcher = SentencePairMatch(model_name="roberta")
    print (matcher.best_concept_match(infos, concepts))

def test_fun_2():
    concepts = ["人物","演员","娱乐人物","歌手","音乐人物","制作人","填词人","词作人"]
    infos = ["刘德华（Andy Lau），1961年9月27日出生于中国香港，演员、歌手、作词人、制片人，香港四大天王成员之一。",
             "1981年出演电影处女作《彩云曲》。",
             "1983年主演的武侠剧《神雕侠侣》在香港获得62点的收视纪录。",
             "1991年创办天幕电影公司。",
             "1992年，凭借传记片《五亿探长雷洛传》获得第11届香港电影金像奖最佳男主角提名。",
             "1994年担任剧情片《天与地》的制片人。",
             "2000年凭借警匪片《暗战》获得第19届香港电影金像奖最佳男主角奖。",
             "2004年凭借警匪片《无间道3：终极无间》获得第41届台湾金马奖最佳男主角奖。",
             "2005年获得香港UA院线颁发的全港最高累积票房香港男演员”奖。",
             "2006年获得釜山国际电影节亚洲最有贡献电影人奖。",
             "2011年主演剧情片《桃姐》，并凭借该片先后获得台湾金马奖最佳男主角奖、香港电影金像奖最佳男主角奖；同年担任第49届台湾电影金马奖评审团主席。",
             "2017年主演警匪动作片《拆弹专家》。",
             "1985年发行首张个人专辑《只知道此刻爱你》。",
             "1990年凭借专辑《可不可以》在歌坛获得关注。",
             "1994年获得十大劲歌金曲最受欢迎男歌星奖。",
             "1995年在央视春晚上演唱歌曲《忘情水》。",
             "2000年被《吉尼斯世界纪录大全》评为“获奖最多的香港男歌手”。",
             "2004年第六次获得十大劲歌金曲最受欢迎男歌星奖。",
             "2016年参与填词的歌曲《原谅我》正式发行。",
             "1994年创立刘德华慈善基金会。",
             "2000年被评为世界十大杰出青年。",
             "2005年发起亚洲新星导计划。",
             "2008年被委任为香港非官守太平绅士。",
             "2016年连任中国残疾人福利基金会副理事长。"]
    matcher = SentencePairMatch(model_name="roberta")
    print (matcher.best_concept_match(infos, concepts))

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    test_fun_1()