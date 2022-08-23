from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))


import os
import platform
import paddle
import paddle.distributed as dist
from visualdl import LogWriter
from paddle import nn
import numpy as np
import random
import time
from collections import OrderedDict
from utils.misc import AverageMeter
from utils import logger
from utils.logger import init_logger

from swin_transformer import *
from loss import  *
from metrics import TopkAcc
from utils.save_load import load_dygraph_pretrain
from utils.save_load import init_model
import utils.save_load as save_load
from preprocess.data_loader import *


@paddle.no_grad()
def eval(eval_dataloader, model, epoch_id=0):
    # set seed
    seed = False
    if seed:
        assert isinstance(seed, int), "The 'seed' must be a integer!"
        paddle.seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # init train_func and eval_func

    # build loss
    eval_loss_func = CELoss()

    # build metric
    eval_metric_func = TopkAcc(topk=[1, 5])
    model.eval() 

    output_info = dict()
    time_info = {
        "batch_cost": AverageMeter(
            "batch_cost", '.5f', postfix=" s,"),
        "reader_cost": AverageMeter(
            "reader_cost", ".5f", postfix=" s,"),
    }
    print_batch_step = 10

    metric_key = None
    tic = time.time()
    max_iter = len(eval_dataloader)
    for iter_id, batch in enumerate(eval_dataloader):
        if iter_id >= max_iter:
            break
        if iter_id == 5:
            for key in time_info:
                time_info[key].reset()

        time_info["reader_cost"].update(time.time() - tic)
        batch_size = batch[0].shape[0]
        batch[0] = paddle.to_tensor(batch[0]).astype("float32")
        batch[1] = batch[1].reshape([-1, 1]).astype("int64")
        # image input
        out = model(batch[0])
        # calc loss
        if eval_loss_func is not None:
            loss_dict = {}
            loss = eval_loss_func(out, batch[1])
            loss = {key: loss[key] * 1 for key in loss}
            loss_dict.update(loss)
            loss_dict["loss"] = paddle.add_n(list(loss_dict.values()))
            for key in loss_dict:
                if key not in output_info:
                    output_info[key] = AverageMeter(key, '7.5f')
                output_info[key].update(loss_dict[key].numpy()[0], batch_size)
        # calc metric
        if eval_metric_func is not None:
            metric_dict = OrderedDict()
            metric_dict.update(eval_metric_func(out, batch[1]))
            if paddle.distributed.get_world_size() > 1:
                for key in metric_dict:
                    paddle.distributed.all_reduce(
                        metric_dict[key], op=paddle.distributed.ReduceOp.SUM)
                    metric_dict[key] = metric_dict[
                        key] / paddle.distributed.get_world_size()
            for key in metric_dict:
                if metric_key is None:
                    metric_key = key
                if key not in output_info:
                    output_info[key] = AverageMeter(key, '7.5f')

                output_info[key].update(metric_dict[key].numpy()[0],
                                        batch_size)

        time_info["batch_cost"].update(time.time() - tic)

        if iter_id % print_batch_step == 0:
            time_msg = "s, ".join([
                "{}: {:.5f}".format(key, time_info[key].avg)
                for key in time_info
            ])

            ips_msg = "ips: {:.5f} images/sec".format(
                batch_size / time_info["batch_cost"].avg)

            metric_msg = ", ".join([
                "{}: {:.5f}".format(key, output_info[key].val)
                for key in output_info
            ])
            logger.info("[Eval][Epoch {}][Iter: {}/{}]{}, {}, {}".format(
                epoch_id, iter_id,
                len(eval_dataloader), metric_msg, time_msg, ips_msg))

        tic = time.time()

    metric_msg = ", ".join([
        "{}: {:.5f}".format(key, output_info[key].avg) for key in output_info
    ])
    logger.info("[Eval][Epoch {}][Avg]{}".format(epoch_id, metric_msg))

    # return 1st metric in the dict
    eval_result = output_info[metric_key].avg

    model.train()
    return eval_result

if __name__ == "__main__":
    # init logger
    pretrained_model="./model_file/SwinTransformer_tiny_patch4_window7_224_pretrained"#.pdparams
    output_dir = './output/'
    log_file = os.path.join(output_dir, "winTransformer_tiny_patch4_window7_224",
                            f"eval.log")
    init_logger(name='root', log_file=log_file)
    # set device
    device = paddle.set_device('gpu')
    logger.info('train with paddle {} and device {}'.format(
        paddle.__version__, device))

    # build dataloader
    eval_dataloader = build_eval_dataloader(device)
    # build model
    model = SwinTransformer_tiny_patch4_window7_224(class_num=1000)
    if pretrained_model is not None:
        load_dygraph_pretrain(
                model, pretrained_model)
    # for distributed
    distributed = paddle.distributed.get_world_size() != 1
    if distributed:
        dist.init_parallel_env()
    if distributed:
        model = paddle.DataParallel(model)
    eval(eval_dataloader, model)
