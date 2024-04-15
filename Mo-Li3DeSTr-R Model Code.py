#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import os
import mmcv
from torch.utils.data import DataLoader
from mmdet3d.core.bbox import CameraInstance3DBoxes
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.apis import train_model
from mmcv.runner import get_dist_info, init_dist
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet3d.datasets.builder import DATASETS
from mmdet3d.datasets.pipelines import Compose

# Dataset registration
@DATASETS.register_module()
class SHIFTDataset(Custom3DDataset):
    CLASSES = ("car", "bicycle", "pedestrian")

    def __init__(self, data_root, ann_file, pipeline, classes=None, test_mode=False):
        super().__init__(data_root, ann_file, '', pipeline, classes=classes, test_mode=test_mode)
        self.data_infos = self.load_annotations(ann_file)

    def load_annotations(self, ann_file):
        data_infos = mmcv.load(os.path.join(self.data_root, ann_file))
        return data_infos

    def get_data_info(self, idx):
        info = self.data_infos[idx]
        lidar_path = os.path.join(self.data_root, info['lidar_path'])
        pcd = o3d.io.read_point_cloud(lidar_path)
        points = np.asarray(pcd.points)
        return points, info['annos']

    def prepare_train_data(self, idx):
        points, annos = self.get_data_info(idx)
        points = points.astype(np.float32)
        
        bboxes = np.stack([anno['bbox_3d'] for anno in annos], axis=0).astype(np.float32)
        labels = np.array([self.CLASSES.index(anno['category_id']) for anno in annos], dtype=np.long)

        gt_bboxes_3d = CameraInstance3DBoxes(bboxes, box_dim=7, origin=(0.5, 0.5, 0.5))
        
        results = {'points': points, 'gt_bboxes_3d': gt_bboxes_3d, 'gt_labels_3d': labels}
        if self.pipeline:
            results = self.pipeline(results)
        return results


#  Transformer model's architecture

import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the Transformer Model
class LiDARTransformer(nn.Module):
    def __init__(self, num_classes, dim_model, num_heads, num_encoder_layers, hidden_dim, num_proposals):
        super(LiDARTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.embedding = RobustPointNetEmbeddingLayer(dim_model)
        self.positional_encoding = PositionalEncoding3D(dim_model)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(dim_model, num_heads, hidden_dim)
            for _ in range(num_encoder_layers)
        ])
        self.object_detection_head = ObjectDetectionHead(dim_model, num_classes, num_proposals)

    def forward(self, x):
        x = x.transpose(1, 2)  # Correct dimension ordering for processing
        x = self.embedding(x)
        x = self.positional_encoding(x)

        for layer in self.encoder_layers:
            x = layer(x, None)  # Assume no mask for simplicity

        detection_output = self.object_detection_head(x)
        return detection_output

class PositionalEncoding3D(nn.Module):
    def __init__(self, dim_model, max_len=5000):
        super(PositionalEncoding3D, self).__init__()
        self.d_model = dim_model
        pe = torch.zeros(max_len, dim_model)
        for pos in range(max_len):
            for i in range(0, dim_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/dim_model)))
                if i + 1 < dim_model:
                    pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * i)/dim_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x

class RobustPointNetEmbeddingLayer(nn.Module):
    def __init__(self, output_dim):
        super(RobustPointNetEmbeddingLayer, self).__init__()
        self.tnet = TNet(k=3)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, output_dim, 1)  # Adjusted output dimension
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = x[:, :3, :]  # Use only x, y, z for TNet
        trans = self.tnet(x)
        x = x.transpose(1, 2).bmm(trans).transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = nn.MaxPool1d(x.size(-1))(x)
        x = x.view(-1, self.bn3.num_features)
        return x

class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.initialize_weights()

    def initialize_weights(self):
        self.fc3.weight.data.fill_(0)
        self.fc3.bias.data = torch.eye(self.k).flatten()

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = nn.MaxPool1d(x.size(-1))(x)
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        iden = torch.eye(self.k, device=x.device).view(1, self.k*self.k).repeat(batch_size, 1)
        x = x + iden
        x = x.view(batch_size, self.k, self.k)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, hidden_dim)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        attention_output = self.attention(x, x, x, mask)
        x = self.norm1(x + attention_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, v, k, q, mask):
        batch_size = q.size(0)

        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        # Scaled dot-product attention
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        dk = torch.tensor(self.depth, dtype=torch.float32)
        scaled_attention_logits = matmul_qk / torch.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = F.softmax(scaled_attention_logits, dim=-1)

        output = torch.matmul(attention_weights, v)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, -1, self.d_model)
        output = self.dense(output)

        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class ObjectDetectionHead(nn.Module):
    def __init__(self, d_model, num_classes, num_proposals):
        super(ObjectDetectionHead, self).__init__()
        self.num_proposals = num_proposals
        self.num_classes = num_classes
        self.d_model = d_model

        self.conv1 = nn.Conv1d(d_model, 256, 1)
        self.conv2 = nn.Conv1d(256, 128, 1)
        self.adaptive_pool = nn.AdaptiveMaxPool1d(1)  # Ensure the output length is 1

        self.bbox_pred = nn.Conv1d(128, num_proposals * 4, 1)  # Outputs num_proposals * 4 features
        self.class_score = nn.Conv1d(128, num_proposals * num_classes, 1)  # Outputs num_proposals * num_classes features

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.adaptive_pool(x)  # This will pool the data to ensure the length dimension is 1

        bbox = self.bbox_pred(x).view(x.size(0), self.num_proposals, 4)  # Reshape to (batch, num_proposals, 4)
        scores = self.class_score(x).view(x.size(0), self.num_proposals, self.num_classes)  # Reshape to (batch, num_proposals, num_classes)
        return bbox, scores

# Training Stage
def main():
    model = LiDARTransformer(num_classes=12, dim_model=2048, num_heads=16, num_encoder_layers=8, hidden_dim=2048, num_proposals=150)
    dataset = SHIFTDataset(
        data_root='D:/Shift Dataset_lidar_data_training',
        ann_file='train.json',
        pipeline=[dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=3, use_dim=3),
                  dict(type='DefaultFormatBundle3D', class_names=('car', 'bicycle', 'pedestrian')),
                  dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])],
        classes=('car', 'bicycle', 'pedestrian')
    )
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(10):  # Runs for 10 epochs
        for i, data in enumerate(data_loader):
            points = data['points'].to(device)
            gt_bboxes_3d = data['gt_bboxes_3d'].to(device)
            gt_labels_3d = data['gt_labels_3d'].to(device)
            
            optimizer.zero_grad()
            predictions = model(points)
            loss = compute_loss(predictions, gt_bboxes_3d, gt_labels_3d)  # Define your loss computation
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch + 1}/{10}], Step [{i}], Loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()


# In[ ]:


#Testing Stage

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score

# Load the full model
model = torch.load('Mo-Li3DeSTr-R_Full_Model.pth')
model.eval()  # Prepare the model for evaluation

class SHIFTDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.npz')]
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        point_cloud = torch.tensor(data['point_cloud'], dtype=torch.float32)
        labels = torch.tensor(data['labels'], dtype=torch.long)
        return point_cloud, labels

def test_model(model, device, data_loader):
    model.to(device)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    average_precision = average_precision_score(all_labels, all_preds, average='macro')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Average Precision: {average_precision:.4f}')

# Main function to execute testing
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data_path = 'D:\Shift Dataset_lidar_data_testing'  # Path to the test dataset
    test_dataset = SHIFTDataset(root_dir=test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    test_model(model, device, test_loader)

if __name__ == "__main__":
    main()

