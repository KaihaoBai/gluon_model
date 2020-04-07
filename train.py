# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import division

import argparse
import logging
import time

from config import *

import numpy as np

import mxnet as mx
#from data import csv_iterator
from mxnet import gluon
from mxnet.gluon.model_zoo import vision as models
from mxnet import autograd as ag, nd
from model import *

parser = argparse.ArgumentParser(description='train dcn')
parser.add_argument('--num-epoch', type=int, default=2)
parser.add_argument('--batch-size', type=int, default=2048)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--gpu-num', type=int, default=1)
parser.add_argument('--optimizer', type=str, default='adam')

if __name__ == '__main__':    
    args = parser.parse_args()
    logging.info(args)
    num_epoch = args.num_epoch
    batch_size = args.batch_size
    optimizer = args.optimizer
    lr = args.lr
    gpu_num = args.gpu_num


    train_data = mx.io.CSVIter(data_csv="/home/khbai/env_examples/Deep-Cross-Model/data/train.csv.data1.csv", data_shape= (CRITEO_FIELD_NUM,), \
                                   label_csv="/home/khbai/env_examples/Deep-Cross-Model/data/train.csv.label1.csv", label_shape = (1,), \
                                   batch_size=batch_size, round_batch=False,\
                                   prefetching_buffer=1)

    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    embeddings = []
    for i, num in enumerate(CRITEO_FEATURE_NUM):
        embed = gluon.nn.Embedding(num, 80)
        embed.initialize(mx.init.Uniform(), ctx=[mx.cpu()])
        embeddings.append(embed)
    net = ConcatNet(embeddings)
    weight_dims = CRITEO_FIELD_NUM * 80
    net1 = DeepNet()
    ctx=[mx.gpu(i) for i in range(gpu_num)]
    ctx.append(mx.cpu())
    net1.initialize(mx.init.Uniform(), ctx=ctx)
    net2 = CrossNet(weight_dims)
    optim = mx.optimizer.create(optimizer, learning_rate=lr, rescale_grad=1.0/batch_size, lazy_update=False)
    kvstore1 = mx.kvstore.create('device')
    kvstore2 = mx.kvstore.create('device')
    trainer1 = gluon.Trainer(net.collect_params(), optim, kvstore=kvstore1)
    trainer2 = gluon.Trainer(net1.collect_params(), optim, kvstore=kvstore2)
    acc = mx.metric.Accuracy()
    for epoch in range(num_epoch):
        batch = train_data.next()
        while (batch != ''):
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=[mx.cpu()], batch_axis=0)
            label = batch.label[0]
            label = nd.array(label)
            Ls = []
            with mx.autograd.record():
                for gpu_id in range(gpu_num):
                    for x, _ in zip(data, label):
                        x = net(x)
                    
                        x = x.copyto(mx.gpu(gpu_id))
                        y = net1(x)
                        label = label.copyto(mx.gpu(gpu_id))
                        with mx.autograd.pause():
                            acc.update([label], [y])
                        l = loss(y, label)
                        Ls.append(l)
                print (acc.get())
                for l in Ls:
                    l.backward()
            trainer1.step(batch.data[0].shape[0])
            trainer2.step(batch.data[0].shape[0], ignore_stale_grad=True)
            batch = train_data.next()
            
