import torch
from torch import nn

import math

class EarlyFusionNetwork(nn.Module):

    def __init__(self, hyperparms=None):

        super(EarlyFusionNetwork, self).__init__()        
        self.dropout = nn.Dropout(0.3)
        self.vision_projection = nn.Linear(512, 256) 
        self.text_projection = nn.Linear(768, 256)
        self.fc1 = nn.Linear(256, 256) 
        self.bn1 = nn.BatchNorm1d(256)
        self.classifier = nn.Linear(256, 1) 
        W = torch.Tensor(256, 256)
        self.W = nn.Parameter(W)
        self.relu_f = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # initialize weight matrices
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        
    def forward(self, image_emb, text_emb):

        x1 = image_emb
        x1 = torch.nn.functional.normalize(x1, p=2, dim=1)
        Xv = self.relu_f(self.vision_projection(x1))
        
        x2 = text_emb
        x2 = torch.nn.functional.normalize(x2, p=2, dim=1)
        Xt = self.relu_f(self.text_projection(x2))
        
        
        Xvt = Xv * Xt
        Xvt = self.relu_f(torch.mm(Xvt, self.W.t()))

        Xvt = self.fc1(Xvt)
        Xvt = self.bn1(Xvt)
        Xvt = self.dropout(Xvt)
        Xvt = self.classifier(Xvt)
        Xvt = self.sigmoid(Xvt)
        return Xvt