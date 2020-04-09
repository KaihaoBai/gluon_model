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


from mxnet import gluon, nd
from mxnet.gluon import nn, Block, HybridBlock
import numpy as np

class ConcatNet(HybridBlock):
    def __init__(self,nets,**kwargs):
        super(ConcatNet,self).__init__(**kwargs)
        self.net = []
        for i, net in enumerate(nets):
            self.net.append(net)
            self.register_child(net, 'embedding' + str(i))
    def hybrid_forward(self,F,x):
        y = nd.op.split(x, axis=1, num_outputs=39)
        return F.concat(*[nd.op.squeeze(self.net[i](y[i]), axis=1) for i, x1 in enumerate(y)], dim=1)

class CrossNet(HybridBlock):
    def __init__(self,weight_dims,weight,bias,**kwargs):
        super(CrossNet,self).__init__(**kwargs)
        self.weight = weight
        self.bias = bias
    def hybrid_forward(self,F,x):
        return F.broadcast_add(F.broadcast_mul(x, self.weight), self.bias)

class DeepNet(HybridBlock):
    def __init__(self,**kwargs):
        super(DeepNet,self).__init__(**kwargs)
        self.dense0 = nn.Dense(1024, activation='relu')
        self.dense1 = nn.Dense(512, activation='relu')
        self.dense2 = nn.Dense(256, activation='relu')
        self.dense3 = nn.Dense(126, activation='relu')
    def hybrid_forward(self,F,x):
        return self.dense3(self.dense2(self.dense1(self.dense0(x))))

class CrossDeepNet(HybridBlock):
    def __init__(self, cross, deep, dense,**kwargs):
        super(CrossDeepNet,self).__init__(**kwargs)
        self.cross = cross
        self.deep = deep
        self.dense = dense
    def hybrid_forward(self,F,x):
        return self.dense(F.concat(*[self.cross(x), self.deep(x)], dim=1))
