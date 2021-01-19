# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from eval_matching.image_text_matching_model import ImageTextMatchingModel
from eval_matching.image_text_matching_data import ImageTextMatchingDataset, ImageTextMatchingTorchDataset, ImageTextMatchingEvaluator
from embedding_matching import SentencePairMatch
from lxrt.entry import LXRTEncoder

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


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

        self.model = LXRTEncoder(args, max_seq_length=20, mode='xlr')

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
            ques_id, feats, boxes, sent = datum_tuple[:4]   # avoid handling target
            print (sent)
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                (lang_output, visn_output), pooled_output = self.model(sent, (feats, boxes))
                
                candidate_nums = visn_output.shape[1]
                print (pooled_output.shape)
                pooled_output = pooled_output.unsqueeze(1)
                print (pooled_output.shape)
                pooled_output = pooled_output.repeat(1,candidate_nums,1)
                print (visn_output.shape)
                print (pooled_output.shape)
                sim = torch.cosine_similarity(visn_output.cuda(), pooled_output.cuda(), dim=2)
                # print ("sim=", sim)
                cos_score = torch.max(sim, dim=1)
                print ("cos_score=", cos_score)

                #print (score.shape)
                #print (predict)
                for qid, l in zip(ques_id.tolist(), cos_score.values.cpu().numpy()):
                    quesid2ans[qid] = str(l)
                print (quesid2ans)
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        dset, loader, evaluator = eval_tuple
        quesid2ans = self.predict(eval_tuple, dump)
        return evaluator.evaluate(quesid2ans)

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
                get_tuple('minival', bs=950,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'minival_predict.json')
            )
            print(result)
        else:
            assert False, "No such test option for %s" % args.test
    else:
        print('Splits in Train data:', matching.train_tuple.dataset.splits)
        if matching.valid_tuple is not None:
            print('Splits in Valid data:', matching.valid_tuple.dataset.splits)
        else:
            print("DO NOT USE VALIDATION")
        matching.train(matching.train_tuple, matching.valid_tuple)


