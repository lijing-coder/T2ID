import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import random
from collections import deque
import numpy as np
import os
import math
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

######STE########
class BinaryGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class FeedForward_MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.3):
        super(FeedForward_MLP, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.ffn(x))

class Confidence_Classification_SubNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.feature_extractor = FeedForward_MLP(input_dim, hidden_dim, dropout)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)  # shape: [B, input_dim]
        logits = self.classifier(x)    # shape: [B, num_classes]
        return logits

class T2ID(nn.Module):
    def __init__(self, num_classes=5, p_missing=0.4, memory_size=500, init_top_k=5, max_top_k=20, dimension=2048, pretrained_backbone=False):
        super(T2MD, self).__init__()
        self.p_missing = p_missing
        self.memory_size = memory_size
        self.temperature = 0.07
        self.memory_bank = deque(maxlen=memory_size)

        resnet50_fundus = torchvision.models.resnet50(weights='DEFAULT' if pretrained_backbone else None)
        resnet50_oct = torchvision.models.resnet50(weights='DEFAULT' if pretrained_backbone else None)
        self.fundus_branch = nn.Sequential(*list(resnet50_fundus.children())[:-1])
        self.oct_branch = nn.Sequential(*list(resnet50_oct.children())[:-1])

        self.FeedForward_fundus = FeedForward_MLP(dimension, int(dimension*0.5), dropout=0.5)
        self.FeedForward_oct = FeedForward_MLP(dimension, int(dimension*0.5), dropout=0.5)
        self.FeedForward_fusion = FeedForward_MLP(dimension*2, dimension, dropout=0.5)

        #Confidence estimation
        self.Confidence_fundus = Confidence_Classification_SubNetwork(dimension, int(dimension*0.5), 1)
        self.Confidence_oct = Confidence_Classification_SubNetwork(dimension, int(dimension*0.5), 1)
        self.Confidence_fusion = Confidence_Classification_SubNetwork(dimension*2, dimension, 1)

        #Classification
        self.Classification_fundus = Confidence_Classification_SubNetwork(dimension, int(dimension*0.5), num_classes)
        self.Classification_oct = Confidence_Classification_SubNetwork(dimension, int(dimension*0.5), num_classes)
        self.Classification_fusion = Confidence_Classification_SubNetwork(dimension*2, dimension, num_classes)

        self.weight1 = nn.Parameter(torch.tensor(1.0))

        self.gate_network_1 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        self.gate_network_2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def _retrieve_cross_modal(self, query_feature, query_modality='fundus'):
        if len(self.memory_bank) == 0:
            return torch.zeros_like(query_feature)

        fundus_feats = torch.stack([pair[0] for pair in self.memory_bank])
        oct_feats = torch.stack([pair[1] for pair in self.memory_bank])

        if query_modality == 'fundus':
            query_set = fundus_feats
            target_set = oct_feats
        elif query_modality == 'oct':
            query_set = oct_feats
            target_set = fundus_feats
        else:
            raise ValueError("query_modality must be 'fundus' or 'oct'")

        similarities = F.cosine_similarity(query_feature.unsqueeze(0), query_set, dim=1)
        
        mask = similarities > 0.5
        filtered_similarities = similarities[mask]
        filtered_indices = torch.nonzero(mask, as_tuple=False).squeeze()

        if filtered_indices.ndim == 0:
            filtered_indices = filtered_indices.unsqueeze(0)
        if filtered_similarities.numel() == 0:
            return torch.zeros_like(query_feature)
            
        mean = filtered_similarities.mean()
        std = filtered_similarities.std(unbiased=False)
        alpha = 0.5
        dynamic_threshold = mean - alpha * std
        final_mask = filtered_similarities > dynamic_threshold
        final_similarities = filtered_similarities[final_mask]
        final_indices = filtered_indices[final_mask]

        if final_similarities.numel() == 0:
            closest_idx = torch.argmin(torch.abs(similarities - mean))
            weighted_feature = target_set[closest_idx]
            return weighted_feature

        topk_target_feats = target_set[final_indices]
        weights = F.softmax(final_similarities, dim=0)
        weighted_feature = torch.sum(weights.view(-1, 1) * topk_target_feats, dim=0)
        #print(f"[INFO] Selected k: {final_similarities.numel()} | Mean sim: {mean.item():.3f} | Std: {std.item():.3f} | Threshold: {dynamic_threshold.item():.3f}")

        return weighted_feature

    def confidence_loss(self, TCPLogit, TCPConfidence, label):
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        pred = F.softmax(TCPLogit, dim=1)
        p_target = torch.gather(input=pred, dim=1, index=label.cuda().unsqueeze(dim=1).type(torch.int64)).view(-1)
        c_loss = F.mse_loss(TCPConfidence.view(-1), p_target, reduction='none')
        return c_loss

    def forward(self, fundus_img, oct_img, label, update_memory=False):

        original_missing_features = []
        reconstructed_features = []
        w1 = torch.exp(self.weight1)
        batch_size = fundus_img.size(0)

        fundus_features = torch.flatten(self.fundus_branch(fundus_img),1)
        oct_features = torch.flatten(self.oct_branch(oct_img),1)

        fundus_features, oct_features, missing_mask, mask1, mask2 = apply_missing_fixed(fundus_features, oct_features, p_missing=self.p_missing, mode='alternate', return_mask=True)

        for i in range(batch_size):
            if missing_mask[i, 0] == 1:
                existing_feature = oct_features[i]
                gate_val = self.gate_network_1(existing_feature)
                gate_decision = BinaryGate.apply(gate_val)

                if gate_decision > 0.5 and len(self.memory_bank) > 0:
                    filled_feature = self._retrieve_cross_modal(
                        existing_feature.detach(), query_modality='oct'
                    )
                    fundus_output[i] = filled_feature
                    original_missing_features.append(fundus_features[i].detach().clone())
                    reconstructed_features.append(filled_feature.detach().clone())
                else:
                    fundus_output[i] = torch.zeros_like(fundus_features[i])

            elif missing_mask[i, 1] == 1:
                existing_feature = fundus_features[i]
                gate_val = self.gate_network_2(existing_feature)
                gate_decision = BinaryGate.apply(gate_val)

                if gate_decision > 0.5 and len(self.memory_bank) > 0:
                    filled_feature = self._retrieve_cross_modal(
                        existing_feature.detach(), query_modality='fundus'
                    )
                    oct_output[i] = filled_feature
                    original_missing_features.append(oct_features[i].detach().clone())
                    reconstructed_features.append(filled_feature.detach().clone())
                else:
                    oct_output[i] = torch.zeros_like(oct_features[i])
                    
            else:
                if update_memory and self.training:
                    if len(self.memory_bank) < self.memory_size:
                        self.memory_bank.append((fundus_features[i].detach().clone(), oct_features[i].detach().clone()))
                    else:
                        fundus_set = torch.stack([p[0] for p in self.memory_bank])
                        similarity = F.cosine_similarity(fundus_features[i].unsqueeze(0), fundus_set, dim=1)
                        if similarity.max() < 0.8:
                            self.memory_bank.append((fundus_features[i].detach().clone(), oct_features[i].detach().clone()))

        fundus_output = self.FeedForward_fundus(fundus_output)
        oct_output = self.FeedForward_oct(oct_output)

        TCPConfidence_fundus = self.Confidence_fundus(fundus_output)
        TCPConfidence_fundus = torch.sigmoid(TCPConfidence_fundus)
        TCPConfidence_oct = self.Confidence_oct(oct_output)
        TCPConfidence_oct = torch.sigmoid(TCPConfidence_oct)

        TCPLogit_fundus = self.Classification_fundus(fundus_output)
        TCPLogit_oct = self.Classification_oct(oct_output)

        fundus_output = fundus_output * TCPConfidence_fundus
        oct_output = oct_output * TCPConfidence_oct

        feature_fusion = torch.cat([fundus_output, oct_output], dim=1)
        feature_fusion = self.FeedForward_fusion(feature_fusion)
        Confidence_fusion = self.Confidence_fusion(feature_fusion)
        Confidence_fusion = torch.sigmoid(Confidence_fusion)
        feature_fusion = feature_fusion * Confidence_fusion
        Logit_fusion = self.Classification_fusion(feature_fusion)

        c_loss_fundus = self.confidence_loss(TCPLogit_fundus, TCPConfidence_fundus, label)
        c_loss_oct = self.confidence_loss(TCPLogit_oct, TCPConfidence_oct, label)
        c_loss_fusion = self.confidence_loss(Logit_fusion, Confidence_fusion, label)

        c_loss = (c_loss_fundus + c_loss_oct + c_loss_fusion)*w1

        return Logit_fusion, feature_fusion, c_loss
