import os

import numpy as np
import torch
import time

from timm.utils import AverageMeter
from torch import optim, nn
from tqdm import tqdm

from torch.nn import functional as F

from model.modelbase import BaseBackbone
from train.base import TrainBase
from model.clip.optimization import BertAdam
from utils.calc_utils import calc_map_k_matrix

from .get_args import get_args
from .loss import DDSPHLoss

class DDSPHTrainer(TrainBase):

    def __init__(self, args, rank):
        args = get_args(args)
        args.rank = rank
        super(GMLTrainer, self).__init__(args)
        self.logger.info("dataset len: {}".format(len(self.train_loader.dataset)))

        self.run()

    def _init_model(self):
        self.logger.info("init model.")

        self.model = BaseBackbone(outputDim=self.args.output_dim,
                           preload=self.args.preload, writer=self.writer,
                           logger=self.logger, is_train=True).to(self.rank)

        if self.args.pretrained != "" and os.path.exists(self.args.pretrained):
            self.logger.info("load pretrained model.")
            self.model.load_state_dict(torch.load(self.args.pretrained, map_location=f"cuda:{self.rank}"), strict=False)
        self.model.float()

        self.criterion = DDSPHLoss(self.args).to(self.rank)

        to_optim = [{'params': self.model.backbone.parameters(), 'lr': self.args.clip_lr},
                    {'params': self.model.image_hash.parameters(), 'lr': self.args.lr},
                    {'params': self.model.text_hash.parameters(), 'lr': self.args.lr},
                    {'params': self.criterion.parameters(), 'lr': 0.03}
        ]

        self.optimizer = BertAdam(to_optim, lr=self.args.lr, warmup=self.args.warmup_proportion,
                                  schedule='warmup_cosine',b1=0.9, b2=0.98, e=1e-6,
                                  t_total=len(self.train_loader) * self.args.epochs,
                                  weight_decay=self.args.weight_decay, max_grad_norm=1.0)

        self.total_time = 0.0
        print(self.model)

    def train_epoch(self, epoch):
        self.change_state(mode="train")
        self.logger.info(">>>>>> epochs: %d/%d" % (epoch, self.args.epochs))
        stat_meters = {}
        for x in ["L_s", "L_p", "L_t", "loss", "mAP"]:
            stat_meters[x] = AverageMeter()

        for image, text, label, index in tqdm(self.train_loader):
            start_time = time.time()

            image = image.to(self.rank, non_blocking=True).float()
            text = text.to(self.rank, non_blocking=True)
            label = label.to(self.rank, non_blocking=True).float()

            label = self.add_noise_to_labels(label)

            img_embs, txt_embs = self.model(image, text)
            L_s, L_p, L_t = self.criterion(img_embs, txt_embs, label)
            stat_meters["L_s"].update(L_s.item())
            stat_meters["L_p"].update(L_p.item())
            stat_meters["L_t"].update(L_t.item())
            loss = L_s / L_s.detach() + L_p / L_p.detach() + L_t / L_t.detach()

            stat_meters["loss"].update(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.total_time += time.time() - start_time
            torch.cuda.empty_cache()

            map1 = calc_map_k_matrix(img_embs.detach().sign(), txt_embs.detach().sign(), label, label)
            map2 = calc_map_k_matrix(txt_embs.detach().sign(), img_embs.detach().sign(), label, label)
            stat_meters["mAP"].update((map1 + map2) / 2)

            torch.cuda.empty_cache()

        sm_str = ""
        for x in stat_meters.keys():
            sm_str += f"[{x}:{stat_meters[x].avg:.4f}]"
        self.logger.info(
            f">>>>>> [{epoch}/{self.args.epochs}], time: {self.total_time}] {sm_str}")