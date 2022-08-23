# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

import platform
import paddle
import paddle.distributed as dist
from visualdl import LogWriter
from paddle import nn
import numpy as np
import random
import time
from utils.utils import  log_info
from utils.misc import AverageMeter
from utils import logger
from utils.logger import init_logger
from swin_transformer import *
from loss import CELoss, MixCELoss
from metrics import TopkAcc
from optimizer.learning_rate import Cosine
from optimizer.optimizer import AdamW
from utils.save_load import load_dygraph_pretrain
from utils.save_load import init_model
import utils.save_load as save_load
import eval as eval_func
from imagenet_dataset import *
from paddle.io import DistributedBatchSampler, BatchSampler, DataLoader
from preprocess.batch_operators import *
from preprocess.data_loader import *

def train(pretrained_model=None):
    # set seed
    seed = False
    if seed:
        assert isinstance(seed, int), "The 'seed' must be a integer!"
        paddle.seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # init logger
    output_dir = './output/'
    log_file = os.path.join(output_dir, "DeiT_tiny_patch16_224",
                            f"train.log")
    init_logger(name='root', log_file=log_file)

    print("****",logger)

    # set device
    device = paddle.set_device('gpu')

    print('train with paddle {} and device {}'.format(
        paddle.__version__, device))
    #logger.info('train with paddle {} and device {}'.format(
    #    paddle.__version__, device))
    # build dataloader
    train_dataloader = build_train_dataloader(device)

    # build dataloader
    eval_dataloader = build_eval_dataloader(device)

    # build loss
    train_loss_func = CELoss(epsilon=0.1)
    eval_loss_func = CELoss()

    # build metric
    train_metric_func = None
    eval_metric_func = TopkAcc(topk=[1, 5])


    # build model
    model = SwinTransformer_tiny_patch4_window7_224(class_num=1000)

    if pretrained_model is not None:
        load_dygraph_pretrain(
                model, pretrained_model)

    # build optimizer
    lr_sch = Cosine(learning_rate=5e-4,
                        step_each_epoch=len(train_dataloader),
                        epochs=300,
                        eta_min=1e-5,
                        warmup_epoch=20,
                        warmup_start_lr=1e-6)()

    #print("build lr ({}) success..".format(lr_sch))                    
    logger.debug("build lr ({}) success..".format(lr_sch))
    optimizer = AdamW(learning_rate=lr_sch,
                            grad_clip=None,
                            beta1=0.9,
                            beta2=0.999,
                            epsilon=1e-8,
                            weight_decay=0.05,
                            no_weight_decay_name='absolute_pos_embed relative_position_bias_table .bias norm',
                            one_dim_param_no_weight_decay=True)([model])
    logger.debug("build optimizer ({}) success..".format(optimizer))
    #print("build optimizer ({}) success..".format(optimizer))

    # for distributed
    distributed = paddle.distributed.get_world_size() != 1
    if distributed:
        dist.init_parallel_env()
    if distributed:
        model = paddle.DataParallel(model)

    print_batch_step = 10
    save_interval = 1
    best_metric = {
        "metric": 0.0,
        "epoch": 0,
    }
    # key:
    # val: metrics list word
    output_info = dict()
    time_info = {
        "batch_cost": AverageMeter(
            "batch_cost", '.5f', postfix=" s,"),
        "reader_cost": AverageMeter(
            "reader_cost", ".5f", postfix=" s,"),
    }
    # global iter counter
    global_step = 0
    max_iter = len(train_dataloader)
    for epoch_id in range(best_metric["epoch"] + 1,
                            300 + 1):
        acc = 0.0
        # for one epoch train
        tic = time.time()
        
        for iter_id, batch in enumerate(train_dataloader):
            if iter_id >= max_iter:
                break
            if iter_id == 5:
                for key in time_info:
                    time_info[key].reset()
            time_info["reader_cost"].update(time.time() - tic)

            batch_size = batch[0].shape[0]
            batch[1] = batch[1].reshape([-1, 1]).astype("int64")
            #print("*****batch[1]",type(batch[1]),type(batch[1:]),batch[1:])
            global_step += 1
            out = model(batch[0])

            # calc loss
            loss_dict = {}

            #print(len(batch[1:]))
            #label = paddle.to_tensor(batch[1:])

            loss = train_loss_func(out, batch[1])
            loss = {key: loss[key] * 1 for key in loss}
            loss_dict.update(loss)
            loss_dict["loss"] = paddle.add_n(list(loss_dict.values()))

            # step opt and lr
            loss_dict["loss"].backward()
            optimizer.step()
            optimizer.clear_grad()
            lr_sch.step()

            # below code just for logging
            # update metric_for_logger
            # update_loss_for_logger
            for key in loss_dict:
                if key not in output_info:
                    output_info[key] = AverageMeter(key, '7.5f')
                output_info[key].update(loss_dict[key].numpy()[0], batch_size)
            time_info["batch_cost"].update(time.time() - tic)
            if iter_id % print_batch_step == 0:
                log_info(lr_sch, output_info, time_info, train_dataloader, batch_size, epoch_id, iter_id)
            tic = time.time()

        metric_msg = ", ".join([
            "{}: {:.5f}".format(key, output_info[key].avg)
            for key in output_info
        ])
        logger.info("[Train][Epoch {}/{}][Avg]{}".format(
            epoch_id, 300, metric_msg))
        output_info.clear()
        
        # eval model and save model if possible
        if epoch_id % 1 == 0:
            acc = eval_func.eval(eval_dataloader, model, epoch_id=epoch_id)
            if acc > best_metric["metric"]:
                best_metric["metric"] = acc
                best_metric["epoch"] = epoch_id
                save_load.save_model(
                    model,
                    optimizer,
                    best_metric,
                    output_dir,
                    model_name="DeiT_tiny_patch16_224",
                    prefix="best_model")
            logger.info("[Eval][Epoch {}][best metric: {}]".format(
                epoch_id, best_metric["metric"]))

            model.train()

        # save model
        if epoch_id % save_interval == 0:
            save_load.save_model(
                model,
                optimizer, {"metric": acc,
                                    "epoch": epoch_id},
                output_dir,
                model_name="DeiT_tiny_patch16_224",
                prefix="epoch_{}".format(epoch_id))
            # save the latest model
            save_load.save_model(
                model,
                optimizer, {"metric": acc,
                                    "epoch": epoch_id},
                output_dir,
                model_name="DeiT_tiny_patch16_224",
                prefix="latest")


if __name__ == "__main__":
    train()