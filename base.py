import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from utils import get_logger, get_summary_writer, add_noise_to_labels

from train.MITH.data import generate_dataset
from dataset.dataloader import dataloader
from torch.utils.data import DataLoader
import time
from utils.calc_utils import calc_map_k_matrix as calc_map_k, calc_recall_at_k, calc_ndcg_at_k_matrix, calc_crc_k_matrix

import scipy.io as scio

from sklearn.manifold import TSNE


class TrainBase(object):

    def __init__(self, args):

        self.args = args
        # Ensure backward compatibility for clip_path parameter
        if not hasattr(self.args, 'clip_path') and hasattr(self.args, 'preload'):
            self.args.clip_path = self.args.preload
        if not hasattr(self.args, 'backbone'):
            self.args.backbone = 'clip'
        if hasattr(self.args, 'rand'):
            self.args.seed = self.args.rand
            self.args.prefix += str(self.args.rand)
            # self.args.save_mat = False
            self.args.save_model = False
        if hasattr(self.args, 'noise_rate') and self.args.noise_rate > 0:
            self.args.prefix += 'r' + str(self.args.noise_rate)

        if self.args.prefix != "":
            args.save_dir += '/'+ self.args.prefix
        os.makedirs(args.save_dir, exist_ok=True)

        self._init_writer()
        self.logger.info(self.args)
        self.rank = self.args.rank  # GPU ID

        self._init_dataset()
        self._init_model()

        self.global_step = 0
        # self.global_step_t = 0
        self.max_mapi2t = 0
        self.max_mapt2i = 0
        self.max_map = 0
        self.best_epoch_i = 0
        self.best_epoch_t = 0
        
        # Early stopping attributes
        self.early_stop_patience = getattr(args, 'early_stop_patience', 10)
        self.early_stop_counter = 0
        self.best_combined_map = 0

        if self.args.calc_crc:
            self.max_crc = 0

        if self.args.calc_recall:
            self.max_recalli2t = [0, 0, 0, 0]
            self.max_recallt2i = [0, 0, 0, 0]

        if self.args.calc_NDCG:
            self.max_ndcgi2t = 0
            self.max_ndcgt2i = 0

        if self.args.save_csv:
            self.loss_records = []  # 存储loss值
            self.map_i2t_records = []  # 存储image->text检索指标
            self.map_t2i_records = []  # 存储text->image检索指标

        if self.args.unseen:
            self.fold_map_i2t = []
            self.fold_map_t2i = []

    def _init_dataset(self):

        self.logger.info(f"init {self.args.dataset} dataset.")

        if hasattr(self.args, 'data_dir') and self.args.data_dir:
            data_dir = self.args.data_dir
            self.logger.info(f"Using custom data directory: {data_dir}")
        else:
            default_data_dirs = {
                'flickr': '/home/yuebai/Data/Dataset/CrossModel/MIRFLICKR-25K',
                'coco': '/home/yuebai/Data/Dataset/CrossModel/MS-COCO',
                'nuswide': '/home/yuebai/Data/Dataset/CrossModel/NUS-WIDE',
                'iapr': '/home/yuebai/Data/Dataset/CrossModel/IAPR',
                'wiki': '/home/yuebai/Data/Dataset/CrossModel/WIKI',
                'nuswide-lt': '/home/yuebai/Data/Dataset/CrossModal_Longtail/NUS-WIDE',
                'flickr-lt': '/home/yuebai/Data/Dataset/CrossModal_Longtail/MIRFLICKR-25K'
            }
            
            if self.args.dataset not in default_data_dirs:
                raise ValueError(f"Unknown dataset: {self.args.dataset}. Available: {list(default_data_dirs.keys())}")
            
            data_dir = default_data_dirs[self.args.dataset]
            self.logger.info(f"Using default data directory: {data_dir}")

        class_config = {
            'flickr': 24,
            'coco': 80,
            'nuswide': 21,
            'iapr': 291,
            'wiki': 10,
            'nuswide-lt': 21,
            'flickr-lt': 24
        }
        self.args.nclass = class_config.get(self.args.dataset, 0)
        if self.args.nclass == 0:
            raise ValueError(f"Unknown dataset for class count: {self.args.dataset}")
        
        # Wiki数据集特殊配置
        if self.args.dataset == 'wiki':
            self.args.train_num = 2173
            self.args.query_num = 693
        
        # 构建文件路径
        self.index_file = os.path.join(data_dir, "index.mat")
        # NUS-WIDE数据集使用txt格式的caption文件
        if 'nuswide' in self.args.dataset:
            self.caption_file = os.path.join(data_dir, "caption.txt")
        else:
            self.caption_file = os.path.join(data_dir, "caption.mat")
        self.label_file = os.path.join(data_dir, "label.mat")

        if self.args.method == 'MITH':
            train_data, query_data, retrieval_data = generate_dataset(captionFile=self.caption_file,
                                                                      indexFile=self.index_file,
                                                                      labelFile=self.label_file,
                                                                      maxWords=self.args.max_words,
                                                                      imageResolution=self.args.resolution,
                                                                      query_num=self.args.query_num,
                                                                      train_num=self.args.train_num,
                                                                      seed=self.args.seed,
                                                                      unseen=self.args.unseen)
        else:
            train_data, query_data, retrieval_data, collate_fn = dataloader(captionFile=self.caption_file,
                                                                indexFile=self.index_file,
                                                                labelFile=self.label_file,
                                                                maxWords=self.args.max_words,
                                                                imageResolution=self.args.resolution,
                                                                query_num=self.args.query_num,
                                                                train_num=self.args.train_num,
                                                                seed=self.args.seed,
                                                                clip=self.args.backbone == "clip",
                                                                wiki=self.args.dataset =="wiki",
                                                                unseen=self.args.unseen)
        if not self.args.unseen:
            self.train_labels = train_data.get_all_label().to(self.rank)
            self.query_labels = query_data.get_all_label()
            self.retrieval_labels = retrieval_data.get_all_label()
            self.retrieval_num = len(self.retrieval_labels)
            self.logger.info(f"train shape: {self.train_labels.shape}")
            self.logger.info(f"query shape: {self.query_labels.shape}")
            self.logger.info(f"retrieval shape: {self.retrieval_labels.shape}")

            self.train_loader = DataLoader(
                dataset=train_data,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True,
                shuffle=True,
                collate_fn=collate_fn,
                drop_last=True
            )
            self.query_loader = DataLoader(
                dataset=query_data,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True,
                shuffle=True,
                collate_fn=collate_fn
            )
            self.retrieval_loader = DataLoader(
                dataset=retrieval_data,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True,
                shuffle=True,
                collate_fn=collate_fn
            )

            self.max_loader = DataLoader(
                dataset=train_data,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True,
                shuffle=True,
                collate_fn=collate_fn
            )

            self.n_samples = self.train_loader.dataset.__len__()
        else:
            self.train_loaders, self.query_loaders, self.retrieval_loaders = [], [], []
            for i in range(len(train_data)):
                train_label = train_data[i].get_all_label().to(self.rank)

                query_label = query_data[i].get_all_label()


                retrieval_label = retrieval_data[i].get_all_label()


                self.logger.info(f"train shape: {train_label.shape}")
                self.logger.info(f"query shape: {query_label.shape}")
                self.logger.info(f"retrieval shape: {retrieval_label.shape}")

                self.train_loaders.append(DataLoader(
                    dataset=train_data[i],
                    batch_size=self.args.batch_size,
                    num_workers=self.args.num_workers,
                    pin_memory=True,
                    shuffle=True,
                ))
                self.query_loaders.append(DataLoader(
                    dataset=query_data[i],
                    batch_size=self.args.batch_size,
                    num_workers=self.args.num_workers,
                    pin_memory=True,
                    shuffle=True,
                ))
                self.retrieval_loaders.append(DataLoader(
                    dataset=retrieval_data[i],
                    batch_size=self.args.batch_size,
                    num_workers=self.args.num_workers,
                    pin_memory=True,
                    shuffle=True,
                ))

    def _init_model(self):

        self.model = None
        self.model_ddp = None

    def _init_writer(self):

        log_name = self.args.dataset + "-" + str(self.args.output_dim) + "-" + ("train.log" if self.args.is_train else "test.log")
        self.logger = get_logger(os.path.join(self.args.save_dir, log_name))
        self.writer = get_summary_writer(os.path.join(self.args.save_dir, "tensorboard"))

    def run(self):

        if not self.args.unseen:
            if self.args.is_train:
                self.train()
            else:
                for epoch in range(5):
                    self.valid(epoch)
        else:
            for i in range(len(self.train_loaders)):
                self.train_unseen()
            self.logger.info(
                f">>>>>>> FINISHED >>>>>> Best epoch, I-T mAP: {self.fold_map_i2t}\n T-I mAP: {self.fold_map_t2i}")

    def change_state(self, mode):
        if mode == "train":
            self.model.train()
            if hasattr(self, "criterion"):
                self.criterion.train()
        elif mode == "valid":
            self.model.eval()
            if hasattr(self, "criterion"):
                self.criterion.eval()

    @torch.no_grad()
    def get_code(self, data_loader, length: int):

        if self.args.method in ["PVSE", 'PCME'] and self.args.num_embeds != 1 :
            img_buffer = torch.empty(length, self.args.num_embeds, self.args.output_dim, dtype=torch.float).to(self.rank)
            text_buffer = torch.empty(length, self.args.num_embeds, self.args.output_dim, dtype=torch.float).to(self.rank)
        else:
            img_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
            text_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
        encoder_time = 0
        for image, text, label, index in tqdm(data_loader):
            start_encoder_time = time.time()
            if self.args.backbone == 'clip':
                image = image.to(self.rank, non_blocking=True)
                text = text.to(self.rank, non_blocking=True)
            index = index.numpy()
            image_hash = self.model.encode_image(image)
            image_hash = torch.sign(image_hash)
            text_hash = self.model.encode_text(text)
            text_hash = torch.sign(text_hash)
            encoder_time = time.time() - start_encoder_time
            img_buffer[index, :] = image_hash.data
            text_buffer[index, :] = text_hash.data

        return img_buffer, text_buffer, encoder_time

    @torch.no_grad()
    def make_hash_code_DCHMT(self, code: list) -> torch.Tensor:

        code = torch.stack(code)
        code = code.permute(1, 0, 2)
        hash_code = torch.argmax(code, dim=-1)
        hash_code[torch.where(hash_code == 0)] = -1
        hash_code = hash_code.float()

        return hash_code

    @torch.no_grad()
    def get_code_DCHMT(self, data_loader, length: int):

        img_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
        text_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)

        for image, text, label, index in tqdm(data_loader):
            start_encoder_time = time.time()
            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            index = index.numpy()
            image_hash = self.model.encode_image(image)
            image_hash = self.make_hash_code_DCHMT(image_hash)
            text_hash = self.model.encode_text(text)
            text_hash = self.make_hash_code_DCHMT(text_hash)
            encoder_time = time.time() - start_encoder_time
            img_buffer[index, :] = image_hash.data
            text_buffer[index, :] = text_hash.data

        return img_buffer, text_buffer, encoder_time

    @torch.no_grad()
    def get_code_MITH(self, data_loader, length: int):

        img_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
        text_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
        encoder_time = 0
        for image, text, key_padding_mask, label, index in tqdm(data_loader):
            start_encoder_time = time.time()
            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            index = index.numpy()
            key_padding_mask = key_padding_mask.to(self.rank, non_blocking=True)
            output_dict = self.model(image, text, key_padding_mask)

            img_hash_cls = output_dict['img_cls_hash'].detach()
            txt_hash_cls = output_dict['txt_cls_hash'].detach()

            img_tokens_hash = output_dict['img_tokens_hash'].detach()
            txt_tokens_hash = output_dict['txt_tokens_hash'].detach()

            img_buffer[index, :] = torch.sign(img_tokens_hash + img_hash_cls)
            text_buffer[index, :] = torch.sign(txt_tokens_hash + txt_hash_cls)
            encoder_time = time.time() - start_encoder_time

        return img_buffer, text_buffer, encoder_time

    def save_model(self):
        save_dir = os.path.join(self.args.save_dir, "model")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_dir, "model.pth"))
        self.logger.info("save mode to {}".format(os.path.join(self.args.save_dir, "model.pth")))

    def train_epoch(self, epoch):
        raise NotImplementedError("Function of 'train' doesn't implement.")

    def train(self):
        """
        执行完整的训练流程
        
        训练流程包括：
        1. 循环执行指定轮数（args.epochs）的训练
        2. 每个epoch调用子类实现的train_epoch方法进行训练
        3. 如果启用验证（args.valid=True）且不在warmup阶段，则进行验证
        4. 记录训练损失和评估指标
        5. 训练结束后保存最佳模型和训练指标
        
        注意：train_epoch方法需要在子类中具体实现，用于定义每个epoch的训练逻辑。
        """
        self.logger.info("Start train.")

        for epoch in range(self.args.epochs):
            # 执行一个epoch的训练
            self.train_epoch(epoch)
            
            # 跳过warmup阶段或未启用验证的情况
            if hasattr(self.args, 'warmup') and epoch < self.args.warmup or not self.args.valid:
                continue
                
            # 执行验证
            self.valid(epoch)
            
            # 检查早停
            if self.early_stop_counter >= self.early_stop_patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs. Best combined map: {self.best_combined_map:.4f}")
                break
            
            # 记录CSV数据（如果启用）
            if self.args.save_csv:
                self.loss_records.append(self.epoch_loss.cpu().numpy())
                self.map_i2t_records.append(self.epoch_i2t.cpu().numpy())
                self.map_t2i_records.append(self.epoch_t2i.cpu().numpy())
                
        # 训练完成，输出最佳结果
        self.logger.info(
            f">>>>>>> FINISHED >>>>>> Best epoch, I-T: {self.best_epoch_i}, mAP: {self.max_mapi2t}, T-I: {self.best_epoch_t}, mAP: {self.max_mapt2i}")
        
        # 保存训练指标到CSV文件（如果启用）
        if self.args.save_csv:
            results_df = pd.DataFrame({
                "Epoch": list(range(1, len(self.loss_records) + 1)),
                "Loss": self.loss_records,
                "MAP_i2t": self.map_i2t_records,
                "MAP_t2i": self.map_t2i_records
            })

            # 保存为CSV
            save_path = os.path.join(self.args.save_dir, "training_metrics.csv")
            results_df.to_csv(save_path, index=False)
            print(f"The indicator has been saved to {save_path}")

    def train_unseen(self):
        self.logger.info("Start train.")
        for i in range(len(self.train_loaders)):
            self.train_loader = self.train_loaders[i]
            self.query_loader = self.query_loaders[i]
            self.retrieval_loader = self.retrieval_loaders[i]
            # 重置早停状态，每个fold独立
            self.early_stop_counter = 0
            self.best_combined_map = 0
            for epoch in range(self.args.epochs):
                self.train_epoch(epoch)
                if hasattr(self.args, 'warmup') and epoch < self.args.warmup or not self.args.valid:
                    continue
                self.valid(epoch)
                # 检查早停
                if self.early_stop_counter >= self.early_stop_patience:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs. Best combined map: {self.best_combined_map:.4f}")
                    break
            self.logger.info(
                f">>>>>>> FINISHED >>>>>> Best epoch, I-T: {self.best_epoch_i}, mAP: {self.max_mapi2t}, T-I: {self.best_epoch_t}, mAP: {self.max_mapt2i}")
            self.fold_map_i2t.append(self.max_mapi2t)
            self.max_mapi2t = 0
            self.fold_map_t2i.append(self.max_mapt2i)
            self.max_mapt2i = 0

    def valid_hook(self):
        pass

    @torch.no_grad()
    def valid(self, epoch=None):
        """
        执行验证评估
        
        验证流程包括：
        1. 将模型切换到评估模式
        2. 为查询集和检索集生成哈希码
        3. 计算跨模态检索指标（mAP, Recall, NDCG等）
        4. 保存最佳模型和哈希码（如果性能提升）
        
        Args:
            epoch: 当前epoch数，用于记录最佳性能对应的epoch
            
        注意：该方法会计算以下指标：
        - mAP (Mean Average Precision): 图像->文本和文本->图像检索的平均精度
        - Recall@K: 前K个结果中的召回率
        - NDCG: 归一化折损累计增益
        - CRC: 跨模态排名一致性
        """
        self.logger.info("Valid.")
        self.change_state(mode="valid")
        topk = 50 if self.args.unseen else None
        self.valid_hook()
        self.query_labels = self.query_loader.dataset.get_all_label()
        self.retrieval_labels = self.retrieval_loader.dataset.get_all_label()
        self.args.query_num = len(self.query_loader.dataset)
        self.retrieval_num = len(self.retrieval_loader.dataset)
        if self.args.method == 'DCHMT':
            query_img, query_txt, q_encoder_time = self.get_code_DCHMT(self.query_loader, self.args.query_num)
            retrieval_img, retrieval_txt, r_encoder_time = self.get_code_DCHMT(self.retrieval_loader,
                                                                               self.retrieval_num)
        elif self.args.method == 'MITH':
            query_img, query_txt, q_encoder_time = self.get_code_MITH(self.query_loader, self.args.query_num)
            retrieval_img, retrieval_txt, r_encoder_time = self.get_code_MITH(self.retrieval_loader,
                                                                              self.retrieval_num)
        else:
            query_img, query_txt, q_encoder_time = self.get_code(self.query_loader, self.args.query_num)
            retrieval_img, retrieval_txt, r_encoder_time = self.get_code(self.retrieval_loader, self.retrieval_num)

        if len(query_img.shape) > 2:
            from train.PVSE._utils import mean_average_precision
            mAPi2t = mean_average_precision(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, topk)
            mAPt2i = mean_average_precision(query_txt, retrieval_img, self.query_labels, self.retrieval_labels, topk)
            mAPi2i = mean_average_precision(query_img, retrieval_img, self.query_labels, self.retrieval_labels, topk)
            mAPt2t = mean_average_precision(query_txt, retrieval_txt, self.query_labels, self.retrieval_labels, topk)
        else:
            mAPi2t = calc_map_k(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, topk, self.rank)
            mAPt2i = calc_map_k(query_txt, retrieval_img, self.query_labels, self.retrieval_labels, topk, self.rank)
            mAPi2i = calc_map_k(query_img, retrieval_img, self.query_labels, self.retrieval_labels, topk, self.rank)
            mAPt2t = calc_map_k(query_txt, retrieval_txt, self.query_labels, self.retrieval_labels, topk, self.rank)

        self.epoch_i2t = mAPi2t
        self.epoch_t2i = mAPt2i

        if self.max_mapi2t < mAPi2t:
            self.best_epoch_i = epoch
            if self.args.save_mat:
                self.save_mat(query_img, query_txt, retrieval_img, retrieval_txt, mode_name="i2t", add=mAPi2t)
            self.max_mapi2t = mAPi2t
        if self.max_mapt2i < mAPt2i:
            self.best_epoch_t = epoch
            if self.args.save_mat:
                self.save_mat(query_img, query_txt, retrieval_img, retrieval_txt, mode_name="t2i", add=mAPt2i)
            self.max_mapt2i = mAPt2i
        if self.max_map < mAPi2t + mAPt2i:
            self.max_map = self.max_mapt2i + self.max_mapi2t
            if self.args.save_model:
                self.save_model()
        
        # Early stopping logic
        current_combined = mAPi2t + mAPt2i
        if current_combined > self.best_combined_map:
            self.best_combined_map = current_combined
            self.early_stop_counter = 0
            self.logger.info(f"Early stopping: Best combined map improved to {self.best_combined_map:.4f}")
        else:
            self.early_stop_counter += 1
            self.logger.info(f"Early stopping: No improvement for {self.early_stop_counter}/{self.early_stop_patience} epochs")
        # if self.args.save_model:
        #     for name, param in self.model.named_parameters():
        #         if torch.isnan(param).any():
        #             print(f"NaN in {name}")
        #             exit()
        #     for name, param in self.criterion.gnn.named_parameters():
        #         if torch.isnan(param).any():
        #             print(f"NaN in {name}")
        #             exit()
        #     for name, param in self.criterion.softmax_classifier.named_parameters():
        #         if torch.isnan(param).any():
        #             print(f"NaN in {name}")
        #             exit()


        self.logger.info(
            f">>>>>> [{epoch}/{self.args.epochs}], MAP(i->t): {mAPi2t}, MAP(t->i): {mAPt2i}, MAP(t->t): {mAPt2t}, MAP(i->i): {mAPi2i}\n"
            f">>>>>> MAX MAP(i->t): {self.best_epoch_i}:{self.max_mapi2t}, MAX MAP(t->i): {self.best_epoch_t}:{self.max_mapt2i}\n"
            f">>>>>> query_encoder_time: {q_encoder_time}, retrieval_encoder_time: {r_encoder_time}")
        if self.args.calc_crc:
            crc = calc_crc_k_matrix(query_img, query_txt, retrieval_img, retrieval_txt, self.query_labels, self.retrieval_labels, K=100)
            if self.max_crc < crc:
                self.max_crc = crc
            self.logger.info(f">>>>>> CRC: {crc} |||| MAX CRC: {self.max_crc}")
        if self.args.calc_NDCG:
            ndcg_i2t = calc_ndcg_at_k_matrix(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, what=1, k=1000)
            ndcg_t2i = calc_ndcg_at_k_matrix(query_txt, retrieval_img, self.query_labels, self.retrieval_labels, what=1, k=1000)
            if self.max_ndcgi2t < ndcg_i2t:
                self.max_ndcgi2t = ndcg_i2t
            if self.max_ndcgt2i < ndcg_t2i:
                self.max_ndcgt2i = ndcg_t2i
            self.logger.info(
                f">>>>>> NDCG(i->t): {ndcg_i2t}, NDCG(t->i): {ndcg_t2i} |||| MAX NDCG(i->t): {self.max_ndcgi2t}, MAX NDCG(t->i): {self.max_ndcgt2i}")

        if self.args.calc_recall:
            recall_i2t = calc_recall_at_k(query_img, retrieval_txt, self.query_labels, self.retrieval_labels,
                                          Ks=[1, 10, 100, 1000])
            recall_t2i = calc_recall_at_k(query_txt, retrieval_img, self.query_labels, self.retrieval_labels,
                                          Ks=[1, 10, 100, 1000])
            for i, j in enumerate((1, 10, 100, 1000)):
                if recall_i2t[j] > self.max_recalli2t[i]:
                    self.max_recalli2t[i] = recall_i2t[j]
                if recall_t2i[j] > self.max_recallt2i[i]:
                    self.max_recallt2i[i] = recall_t2i[j]
            self.logger.info(
                f">>>>>>>Recall @ 1(i->t): {recall_i2t[1]}, Recall @ 10(i->t): {recall_i2t[10]}, Recall @ 100(i->t): {recall_i2t[100]}, Recall @ 1000(i->t): {recall_i2t[1000]}")
            self.logger.info(
                f">>>>>>>Recall @ 1(t->i): {recall_t2i[1]}, Recall @ 10(t->i): {recall_t2i[10]}, Recall @ 100(t->i): {recall_t2i[100]}, Recall @ 1000(t->i): {recall_t2i[1000]}")
            self.logger.info(
                f">>>>>>>Best Recall (i->t): {self.max_recalli2t[0]}, {self.max_recalli2t[1]}, {self.max_recalli2t[2]}, {self.max_recalli2t[3]}")
            self.logger.info(
                f">>>>>>>Best Recall (t->i): {self.max_recallt2i[0]}, {self.max_recallt2i[1]}, {self.max_recallt2i[2]}, {self.max_recallt2i[3]}")

        # 随epoch变动的tsne图    不够有效
        # if self.args.epoch_tsne and epoch % 20 == 0:
        #
        #     tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
        #     query_code = ((query_img + query_txt) / 2).cpu()
        #     query_label = self.query_labels.cpu().numpy()
        #     embeddings = tsne.fit_transform(query_code)
        #
        #     plt.figure(figsize=(10, 8))
        #     num_classes = query_label.shape[1]
        #     colors = plt.get_cmap("tab10", num_classes).colors  # 获取 N 个颜色
        #     query_colors = np.dot(self.query_labels, colors) / np.sum(query_label, axis=1, keepdims=True)
        #
        #     scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=query_colors, alpha=0.6)
        #     plt.colorbar(scatter)
        #     plt.title(f'{self.args.loss} - {self.args.miner} - Epoch {epoch}')
        #     save_path = os.path.join(self.args.save_dir, f'tsne_{self.args.loss}_{self.args.miner}_epoch{epoch}.png')
        #     plt.savefig(save_path)
        #     plt.close()
        #
        # del query_img, query_txt, retrieval_img, retrieval_txt

    @torch.no_grad()
    def test(self, mode_name="i2t"):
        if self.args.pretrained == "":
            raise RuntimeError("test step must load a model! please set the --pretrained argument.")
        self.change_state(mode="valid")
        save_dir = os.path.join(self.args.save_dir, "PR_cruve")
        os.makedirs(save_dir, exist_ok=True)
        if self.args.method == 'DCHMT':
            query_img, query_txt, q_encoder_time = self.get_code_DCHMT(self.query_loader, self.args.query_num)
            retrieval_img, retrieval_txt, r_encoder_time = self.get_code_DCHMT(self.retrieval_loader,
                                                                               self.retrieval_num)
        elif self.args.method == 'MITH':
            query_img, query_txt, q_encoder_time = self.get_code_MITH(self.query_loader, self.args.query_num)
            retrieval_img, retrieval_txt, r_encoder_time = self.get_code_MITH(self.retrieval_loader,
                                                                              self.retrieval_num)
        elif self.args.method in ['DNPH', 'DUaPH']:
            query_img, query_txt, q_encoder_time = self.get_code_DNPH(self.query_loader, self.args.query_num)
            retrieval_img, retrieval_txt, r_encoder_time = self.get_code_DNPH(self.retrieval_loader,
                                                                              self.retrieval_num)
        else:
            query_img, query_txt, q_encoder_time = self.get_code(self.query_loader, self.args.query_num)
            retrieval_img, retrieval_txt, r_encoder_time = self.get_code(self.retrieval_loader, self.retrieval_num)

        mAPi2t = calc_map_k(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPt2i = calc_map_k(query_txt, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPi2i = calc_map_k(query_img, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPt2t = calc_map_k(query_txt, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)

        self.max_mapt2i = max(self.max_mapt2i, mAPt2i)
        self.logger.info(f">>>>>> MAP(i->t): {mAPi2t}, MAP(t->i): {mAPt2i}, MAP(t->t): {mAPt2t}, MAP(i->i): {mAPi2i}")

        query_img = query_img.cpu().detach().numpy()
        query_txt = query_txt.cpu().detach().numpy()
        retrieval_img = retrieval_img.cpu().detach().numpy()
        retrieval_txt = retrieval_txt.cpu().detach().numpy()
        query_labels = self.query_labels.numpy()
        retrieval_labels = self.retrieval_labels.numpy()

        result_dict = {
            'q_img': query_img,
            'q_txt': query_txt,
            'r_img': retrieval_img,
            'r_txt': retrieval_txt,
            'q_l': query_labels,
            'r_l': retrieval_labels
        }
        scio.savemat(
            os.path.join(save_dir, str(self.args.output_dim) + "-ours-" + self.args.dataset + "-" + mode_name + ".mat"),
            result_dict)
        self.logger.info(">>>>>> save all data!")

    def compute_loss(self):
        raise NotImplementedError("Function of 'compute_loss' doesn't implement.")

    def save_mat(self, query_img, query_txt, retrieval_img, retrieval_txt, mode_name="i2t", add=None):
        save_dir = self.args.save_dir
        os.makedirs(save_dir, exist_ok=True)

        query_img = query_img.cpu().detach().numpy()
        query_txt = query_txt.cpu().detach().numpy()
        retrieval_img = retrieval_img.cpu().detach().numpy()
        retrieval_txt = retrieval_txt.cpu().detach().numpy()
        query_labels = self.query_labels.numpy()
        retrieval_labels = self.retrieval_labels.numpy()

        result_dict = {
            'q_img': query_img,
            'q_txt': query_txt,
            'r_img': retrieval_img,
            'r_txt': retrieval_txt,
            'q_l': query_labels,
            'r_l': retrieval_labels
        }

        prefix = f"{self.args.output_dim}-{self.args.method}-{self.args.dataset}-{mode_name}"

        if add is not None:
            # 查找同类文件
            best_existing_add = -float('inf')
            best_existing_path = None

            for fname in os.listdir(save_dir):
                if fname.startswith(prefix) and fname.endswith(".mat"):
                    # 解析已有文件中的add值（假设格式为 xxxx-mode_name-add.mat）
                    match = re.match(rf"{re.escape(prefix)}-(\d+\.?\d*).mat", fname)
                    if match:
                        existing_add = float(match.group(1))
                        if existing_add > best_existing_add:
                            best_existing_add = existing_add
                            best_existing_path = os.path.join(save_dir, fname)

            if add > best_existing_add:
                # 当前add更优，删除旧文件，保存新文件
                if best_existing_path:
                    os.remove(best_existing_path)
                    self.logger.info(f"Deleted old file with lower performance: {best_existing_path}")

                new_path = os.path.join(save_dir, f"{prefix}-{add:.4f}.mat")
                scio.savemat(new_path, result_dict)
                self.logger.info(f">>>>>> Saved improved {mode_name} data with add={add:.4f}!")
            else:
                self.logger.info(
                    f"Skip saving {mode_name} data: add={add:.4f} <= existing best add={best_existing_add:.4f}")
        else:
            # 没有add，直接保存（默认命名）
            path = os.path.join(save_dir, f"{prefix}.mat")
            scio.savemat(path, result_dict)
            self.logger.info(f">>>>>> Saved {mode_name} data without performance check.")

    def calc_learnable_params(self):
        n_parameters = 0
        for x in self.model:
            if x:
                n_parameters += sum(p.numel() for p in x.parameters() if p.requires_grad)
        return n_parameters

    def epoch_hook(self, image, text, label):
        if hasattr(self.args, "noise_rate") and self.args.noise_rate > 0:
                label = add_noise_to_labels(label, self.args.noise_rate).to(self.rank)
                # print("ok")
        return image, text, label


    # def visualize_hash_evolution(self, model, dataloader, epoch, sampling_method):
    #     model.eval()
    #     hash_codes = []
    #     labels = []
    #     with torch.no_grad():
    #         for images, texts, target in dataloader:
    #             img_hash = model.image_module(images.cuda()).sign()  # 假设模型输出[-1,1]
    #             txt_hash = model.text_module(texts.cuda()).sign()
    #             merged_hash = (img_hash + txt_hash) / 2  # 跨模态融合示例
    #             hash_codes.append(merged_hash.cpu())
    #             labels.append(target.cpu())
    #     hash_codes = torch.cat(hash_codes).numpy()
    #     labels = torch.cat(labels).numpy()
    #
    #     # 每隔10个epoch执行t-SNE
    #     if epoch % 10 == 0:
    #         tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    #         embeddings = tsne.fit_transform(hash_codes)
    #
    #         plt.figure(figsize=(10, 8))
    #         scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='tab10', alpha=0.6)
    #         plt.colorbar(scatter)
    #         plt.title(f'{sampling_method} - Epoch {epoch}')
    #         plt.savefig(f'tsne_{sampling_method}_epoch{epoch}.png')
    #         plt.close()

