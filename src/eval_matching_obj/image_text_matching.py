# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections
import json

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from eval_matching_obj.image_text_matching_model import ImageTextMatchingLXRTModel
from eval_matching_obj.image_text_matching_data import ImageTextMatchingDataset, ImageTextMatchingTorchDataset, ImageTextMatchingEvaluator
from embedding_matching import SentencePairMatch
# from lxrt.entry import LXRTEncoder, set_visual_config
from lxrt.modeling import VisualConfig, BertConfig

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')

_DEBUG = False

def get_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = ImageTextMatchingDataset(splits)
    tset = ImageTextMatchingTorchDataset(dset)
    evaluator = ImageTextMatchingEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class ImageTextMatching:
    def __init__(self):
        self.train_tuple = get_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        if args.valid != "":
            valid_bsize = 2048 if args.multiGPU else 512
            self.valid_tuple = get_tuple(
                args.valid, bs=valid_bsize,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None

        #self.model = LXRTEncoder(args, max_seq_length=20, mode='xlr')
        bert_config = BertConfig(vocab_size_or_config_json_file='./model/bert-base-uncased/bert_config.json')
        self.model = ImageTextMatchingLXRTModel(bert_config, args)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.load(args.load_lxmert)

        # GPU options
        if args.multiGPU:
            self.model.multi_gpu()
        self.model = self.model.cuda()

        # Losses and optimizer
        self.mce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(list(self.model.parameters()), args.lr)

        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        for epoch in range(args.epochs):
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, label) in iter_wrapper(enumerate(loader)):
                self.model.train()

                self.optim.zero_grad()
                feats, boxes, label = feats.cuda(), boxes.cuda(), label.cuda()
                logit = self.model(feats, boxes, sent)

                loss = self.mce_loss(logit, label)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, predict = logit.max(1)
                for qid, l in zip(ques_id, predict.cpu().numpy()):
                    quesid2ans[qid] = l

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None):
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}

        matching = SentencePairMatch(model_name="bert", language="en")

        for i, datum_tuple in enumerate(loader):
            #ques_id, feats, boxes, sent, _, _, img_ids = datum_tuple[:7]   # avoid handling target
            ques_id, feats, boxes, sent, _, raw_target, img_ids, gounds = datum_tuple[:8]   # avoid handling target
            # print (sent)
            
            with torch.no_grad():
                cross_relationship_score = self.model(sent, feats, boxes)
                if _DEBUG: print ("cross_relationship_score=", cross_relationship_score.shape) 

                match_labels = torch.max(cross_relationship_score, dim=1)
                if _DEBUG:  print ("cos_score=", match_labels)

                for q_id, i_id, sent_ele, roi_obj, gound, is_match in zip(ques_id.tolist(), img_ids, sent, raw_target, gounds, match_labels.indices.cpu().numpy()):
                    # quesid2ans[qid] = "True" if l == 1 else "False"
                    predict_label = "True" if is_match == 1 else "False"
                    quesid2ans[q_id] = {"answer": predict_label, "image_id": i_id, "sentence": sent_ele, "roi_obj": json.loads(roi_obj), "ground": gound}
                if _DEBUG:  print (quesid2ans)
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        dset, loader, evaluator = eval_tuple
        quesid2ans = self.predict(eval_tuple, dump)
        return evaluator.evaluate_matching(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":
    # Build Class
    matching = ImageTextMatching()

    # Open debug mode
    if args.debug is True:
        _DEBUG = True

    # Load Model
    if args.load is not None:
        matching.load(args.load)

    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False       # Always loading all data in test
        if 'test' in args.test:
            matching.predict(
                get_tuple(args.test, bs=950,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'test_predict.json')
            )
        elif 'val' in args.test:    
            # Since part of valididation data are used in pre-training/fine-tuning,
            # only validate on the minival set.
            result = matching.evaluate(
                #get_tuple('minival', bs=950,
                get_tuple(args.test, bs=950,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'minival_predict.json')
            )
            print (result)
        else:
            assert False, "No such test option for %s" % args.test
    else:
        print('Splits in Train data:', matching.train_tuple.dataset.splits)
        if matching.valid_tuple is not None:
            print('Splits in Valid data:', matching.valid_tuple.dataset.splits)
        else:
            print("DO NOT USE VALIDATION")
        matching.train(matching.train_tuple, matching.valid_tuple)


