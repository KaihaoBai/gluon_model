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
import time
import mxnet as mx
import pdb
from mxnet import gluon
from mxnet.gluon.model_zoo import vision as models
from mxnet import autograd as ag, nd
from model import *
import logging

parser = argparse.ArgumentParser(description='train dcn')
parser.add_argument('--num-epoch', type=int, default=2,
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=2048,
                    help='number of examples per batch')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--gpu-num', type=int, default=1)
parser.add_argument('--optimizer', type=str, default='adam',
                    help='what optimizer to use',
                    choices=["ftrl", "sgd", "adam"])

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)    
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
    ctx1 = ctx + [mx.cpu()]
    net1.initialize(mx.init.Uniform(), ctx=ctx1)
    weight = nd.random.uniform(shape=(1, weight_dims), ctx=mx.gpu(0))
    bias = nd.random.uniform(shape=(weight_dims), ctx=mx.gpu(0))
    net2 = CrossNet(weight_dims, weight, bias)
    net2.initialize(ctx=ctx1)
    dense = nn.Dense(2, activation = 'relu')
    dense.initialize(ctx=ctx1)
    net3 = CrossDeepNet(net1, net2, dense)
    optim = mx.optimizer.create('adam', learning_rate=lr, rescale_grad=1.0/batch_size, lazy_update=False)
    kvstore1 = mx.kvstore.create('device')
    kvstore2 = mx.kvstore.create('device')
    trainer1 = gluon.Trainer(net.collect_params(), optim, kvstore=kvstore1)
    trainer2 = gluon.Trainer(net3.collect_params(), optim, kvstore=kvstore2)
    #acc = mx.metric.Accuracy()
    for epoch in range(num_epoch):
        batch = train_data.next()
        nbatch = 0
        while (batch != ''):
            start = time.time()
            nbatch += 1
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=[mx.cpu()], batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=-1)
            
            Ls = []
            with mx.autograd.record():
                x = net(data[0])
                x = gluon.utils.split_and_load(x, ctx_list=ctx)
                
                for x1, y1 in zip(x, label):
                    Ls = [loss(net3(x1), y1)]
                for l in Ls:
                    l.backward()

            trainer1.step(batch.data[0].shape[0])
            trainer2.step(batch.data[0].shape[0], ignore_stale_grad=True)
            batch = train_data.next()
            
            elapsed = time.time() - start
            logging.info("Epoch [%d]: %f samples / sec"%(epoch, batch_size / elapsed ))
         
