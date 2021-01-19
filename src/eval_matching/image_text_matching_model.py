# coding=utf-8
# Copyleft 2019 project LXRT.

import torch
import torch.nn as nn
from lxrt.tokenization import BertTokenizer
from lxrt.modeling import GeLU, BertLayerNorm, BertPreTrainingHeads, BertPreTrainedModel, LXRTModel, VISUAL_CONFIG
from lxrt.entry import convert_sents_to_features
from param import args


class ImageTextMatchingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lxrt_encoder = LXRTEncoder(args, max_seq_length=20, mode='xlr')
        
        # self.hid_dim = hid_dim = self.lxrt_encoder.dim
        # self.logit_fc = nn.Sequential(
        #     nn.Linear(hid_dim * 2, hid_dim * 2),
        #     GeLU(),
        #     BertLayerNorm(hid_dim * 2, eps=1e-12),
        #     nn.Linear(hid_dim * 2, 2)
        # )
        # self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: b, (string)
        :param leng: b, (numpy, int)
        :return:
        """

        # Extract feature --> Concat
        (lang_output, visn_output), pooled_output = self.lxrt_encoder(sent, (feat, pos))
        inner_product = self.inner_product(lang_output, pooled_output)

        logit = self.logit_fc(inner_product)


        return logit

def set_visual_config(args):
    VISUAL_CONFIG.l_layers = args.llayers
    VISUAL_CONFIG.x_layers = args.xlayers
    VISUAL_CONFIG.r_layers = args.rlayers

class ImageTextMatchingLXRTPretrainedModel(BertPreTrainedModel):
    def __init__(self, bert_config):
        super().__init__(bert_config)
        # Configuration
        self.bert_config = bert_config

        # LXRT backbone
        self.bert = LXRTModel(bert_config)

        # Pre-training heads
        self.cls = BertPreTrainingHeads(bert_config, self.bert.embeddings.word_embeddings.weight)

        # Weight initialization
        self.apply(self.init_bert_weights)

    # 看下论文，为啥这输入有文本，评测二分类也有文本？
    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                feats=None, poses=None, visual_attention_mask=None):
        # print ("poses=", poses.shape)
        (lang_output, visn_output), pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask,
            visual_feats=(feats, poses), 
            visual_attention_mask=None
        )

        lang_prediction_scores, cross_relationship_score = self.cls(lang_output, pooled_output)

        # total_loss = 0.
        # loss_fct = CrossEntropyLoss(ignore_index=-1)
        # losses = ()
        
        # # task_matching
        # if matched_label is not None:
        #     matched_loss = loss_fct(
        #         cross_relationship_score.view(-1, 2),
        #         matched_label.view(-1)
        #     )
        #     total_loss += matched_loss
        #     losses += (matched_loss.detach(),)
        
        # return total_loss, torch.stack(losses).unsqueeze(0)
        return cross_relationship_score


class ImageTextMatchingLXRTModel(BertPreTrainedModel):
    def __init__(self, bert_config, args, max_seq_length = 20):
        super().__init__(bert_config)
        # Configuration
        self.bert_config = bert_config
        set_visual_config(args)

        self.max_seq_length = max_seq_length

        # Using the bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "./model/bert-base-uncased/",
            do_lower_case=True
        )

        # LXRT pretrained model
        self.model = ImageTextMatchingLXRTPretrainedModel(bert_config)


    # 看下论文，为啥这输入有文本，评测二分类也有文本？
    def forward(self, sent, feats, boxes, visual_attention_mask=None):

        train_features = convert_sents_to_features(sent, self.max_seq_length, self.tokenizer)

        input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).cuda()
        segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).cuda()

        feats, boxes = feats.cuda(), boxes.cuda()

        cross_relationship_score = self.model(input_ids=input_ids, 
                                token_type_ids=segment_ids, attention_mask=input_mask, 
                                feats=feats, poses=boxes)

        return cross_relationship_score



    def load(self, path):
        # Load state_dict from snapshot file
        print("Load LXMERT pre-trained model from %s" % path)
        state_dict = torch.load("%s_LXRT.pth" % path)
        # print (state_dict.keys())

        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
        print (state_dict.keys())

        # Print out the differences of pre-trained and model weights.
        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        #cls_keys = set(self.cls.state_dict().keys())
        print()
        print("Weights in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Weights in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        # Load weights to model
        self.model.load_state_dict(state_dict, strict=False)