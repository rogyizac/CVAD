import torch
from torch import nn
import torch.nn.functional as F
import math

# class EarlyFusionNetwork(nn.Module):

#     def __init__(self, hyperparms=None):

#         super(EarlyFusionNetwork, self).__init__()        
#         self.dropout = nn.Dropout(0.3)
#         self.vision_projection = nn.Linear(512, 256) 
#         self.text_projection = nn.Linear(768, 256)
#         self.fc1 = nn.Linear(256, 256) 
#         self.bn1 = nn.BatchNorm1d(256)
#         self.classifier = nn.Linear(256, 1) 
#         W = torch.Tensor(256, 256)
#         self.W = nn.Parameter(W)
#         self.relu_f = nn.LeakyReLU(0.2)
#         self.sigmoid = nn.Sigmoid()
#         # initialize weight matrices
#         nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        
#     def forward(self, image_emb, text_emb):

#         x1 = image_emb
#         x1 = torch.nn.functional.normalize(x1, p=2, dim=1)
#         Xv = self.relu_f(self.vision_projection(x1))
        
#         x2 = text_emb
#         x2 = torch.nn.functional.normalize(x2, p=2, dim=1)
#         Xt = self.relu_f(self.text_projection(x2))
        
        
#         Xvt = Xv * Xt
#         Xvt = self.relu_f(torch.mm(Xvt, self.W.t()))

#         Xvt = self.fc1(Xvt)
#         Xvt = self.bn1(Xvt)
#         Xvt = self.dropout(Xvt)
#         Xvt = self.classifier(Xvt)
#         Xvt = self.sigmoid(Xvt)
#         return Xvt
    
class EarlyFusionNetwork(nn.Module):
    def __init__(self):
        super(EarlyFusionNetwork, self).__init__()
        self.dropout = nn.Dropout(0.3)
        self.vision_projection = nn.Linear(512, 256)
        self.text_projection = nn.Linear(768, 256)
        self.fc1 = nn.Linear(256, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.classifier = nn.Linear(256, 1)
        self.attention = nn.Linear(256, 256)
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)  # Define Softmax here

    def forward(self, image_emb, text_emb):
        # Normalize and project image embeddings
        x1 = F.normalize(image_emb, p=2, dim=1)
        Xv = self.leaky_relu(self.vision_projection(x1))
        
        # Normalize and project text embeddings
        x2 = F.normalize(text_emb, p=2, dim=1)
        Xt = self.leaky_relu(self.text_projection(x2))
        
        # Element-wise multiplication for early fusion
        Xvt = Xv * Xt
        
        # Apply attention
        attention_weights = self.softmax(self.attention(Xvt))  # Use nn.Softmax
        Xvt = Xvt * attention_weights

        # Additional layers
        Xvt = self.fc1(Xvt)
        Xvt = self.bn1(Xvt)
        Xvt = self.leaky_relu(Xvt)
        Xvt = self.dropout(Xvt)

        Xvt = self.fc2(Xvt)
        Xvt = self.bn2(Xvt)
        Xvt = self.leaky_relu(Xvt)
        Xvt = self.dropout(Xvt)

        # Classifier
        Xvt = self.classifier(Xvt)
        Xvt = self.sigmoid(Xvt)
        return Xvt