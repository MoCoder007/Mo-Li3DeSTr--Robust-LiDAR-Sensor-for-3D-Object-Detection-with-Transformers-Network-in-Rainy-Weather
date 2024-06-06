#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial import KDTree


# In[2]:


LABEL_TO_INDEX = {'car': 0, 'bicycle': 1, 'pedestrian': 2}  # Adjusted for your labels
INDEX_TO_LABEL = {0: 'Car', 1: 'Bicycle', 2: 'Pedestrian'}  # Adjusted for your labels

# Reads annotations from a file, converting bounding box information to a format suitable for processing
def read_annotations(file_path):
    annotations = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            class_label = parts[0]
            bbox_center = np.array([float(parts[i]) for i in range(1, 4)])
            bbox_dims = np.array([float(parts[i]) for i in range(4, 7)])
            bbox = np.concatenate([bbox_center - bbox_dims / 2, bbox_center + bbox_dims / 2])  # Convert to [x_min, y_min, z_min, x_max, y_max, z_max]
            annotations.append({
                'class_label': LABEL_TO_INDEX.get(class_label, -1),
                'bbox': bbox
            })
    return annotations

# Filters noise from LiDAR data based on the density of points in the vicinity, keeping points in denser areas.
def filter_noise(lidar_data, k=20, x_threshold=2.0):
    tree = KDTree(lidar_data[:, :3])
    distances, _ = tree.query(lidar_data[:, :3], k=k+1)
    distances = distances[:, 1:]  # Exclude self-match
    mean_distances = np.mean(distances, axis=1)
    std_dev = np.std(mean_distances)
    mean = np.mean(mean_distances)
    filtered_indices = mean_distances < (mean + x_threshold * std_dev)
    return lidar_data[filtered_indices], filtered_indices

# Defines a dataset for loading and preprocessing LiDAR data and annotations for the KITTI dataset.
class KITTIDataset(Dataset):
    def __init__(self, lidar_root_dir, annotation_root_dir, split='train', max_points=120000):
        self.lidar_root_dir = lidar_root_dir
        self.annotation_root_dir = annotation_root_dir
        self.split = split
        self.max_points = max_points
        self.lidar_files = self.get_files(lidar_root_dir, 'velodyne')
        self.annotation_files = self.get_files(annotation_root_dir, 'label_2')

    # Retrieves file paths for the dataset, supporting both LiDAR and annotation data.
    def get_files(self, root_dir, subdir):
        files = []
        path = os.path.join(root_dir, self.split, subdir)
        for file_name in sorted(os.listdir(path)):
            if file_name.endswith('.bin') or file_name.endswith('.txt'):
                files.append(os.path.join(path, file_name))
        return files

    # Returns the number of items in the dataset.
    def __len__(self):
        return len(self.lidar_files)

    # Loads and preprocesses the data for a given index, including noise filtering and label assignment.
    def __getitem__(self, idx):
        preprocessed_data_path = os.path.join(self.lidar_root_dir, f'preprocessed_data_{idx}.npy')
        preprocessed_labels_path = os.path.join(self.lidar_root_dir, f'preprocessed_labels_{idx}.npy')
        
        if os.path.exists(preprocessed_data_path) and os.path.exists(preprocessed_labels_path):
            lidar_data = np.load(preprocessed_data_path)
            class_labels = np.load(preprocessed_labels_path)
        else:
            lidar_data = np.fromfile(self.lidar_files[idx], dtype=np.float32).reshape(-1, 4)
            lidar_data = self.preprocess_lidar_data(lidar_data)
            annotations = read_annotations(self.annotation_files[idx])
            filtered_lidar_data, _ = filter_noise(lidar_data)
            class_labels = self.assign_labels(filtered_lidar_data, annotations)
            np.save(preprocessed_data_path, filtered_lidar_data)
            np.save(preprocessed_labels_path, class_labels)
            lidar_data = filtered_lidar_data
        
        padded_lidar_data, padded_class_labels = self.pad_or_truncate(lidar_data, class_labels)
        return torch.tensor(padded_lidar_data, dtype=torch.float32), torch.tensor(padded_class_labels, dtype=torch.long)

    # Filters LiDAR points based on altitude and distance to keep points within a certain range.
    def preprocess_lidar_data(self, lidar_data):
        z_min, z_max = -1.5, 2.0
        distance_max = 50.0
        distances = np.sqrt(np.sum(lidar_data[:, :3]**2, axis=1))
        altitude_filter = (lidar_data[:, 2] >= z_min) & (lidar_data[:, 2] <= z_max)
        distance_filter = distances <= distance_max
        filtered_indices = altitude_filter & distance_filter
        return lidar_data[filtered_indices]

    # Pads or truncates the LiDAR data and class labels to a fixed size for consistent input size.
    def pad_or_truncate(self, lidar_data, class_labels):
        num_points = lidar_data.shape[0]
        if num_points < self.max_points:
            padding = np.zeros((self.max_points - num_points, lidar_data.shape[1]), dtype=np.float32)
            label_padding = -np.ones(self.max_points - num_points, dtype=np.int64)
            lidar_data = np.vstack((lidar_data, padding))
            class_labels = np.concatenate((class_labels, label_padding))
        elif num_points > self.max_points:
            lidar_data = lidar_data[:self.max_points, :]
            class_labels = class_labels[:self.max_points]
        return lidar_data, class_labels

    # Assigns class labels to LiDAR points based on their location relative to annotated bounding boxes.
    def assign_labels(self, lidar_data, annotations):
        class_labels = np.full(lidar_data.shape[0], -1, dtype=np.int64)
        for annotation in annotations:
            bbox = annotation['bbox']
            in_bbox_indices = np.where(
                (lidar_data[:, 0] >= bbox[0]) & (lidar_data[:, 0] <= bbox[3]) &
                (lidar_data[:, 1] >= bbox[1]) & (lidar_data[:, 1] <= bbox[4]) &
                (lidar_data[:, 2] >= bbox[2]) & (lidar_data[:, 2] <= bbox[5])
            )[0]
            class_labels[in_bbox_indices] = annotation['class_label']
        return class_labels

# Specify your dataset directories
lidar_root_dir = 'D:/data_object_velodyne'
annotation_root_dir = 'D:/data_object_label_2'
kitti_dataset = KITTIDataset(lidar_root_dir, annotation_root_dir)

data_loader = DataLoader(kitti_dataset, batch_size=1, shuffle=True)

# Initialize an empty list to store processed samples
processed_samples = []

# Iterate over the DataLoader

for batch_idx, (data, labels) in enumerate(data_loader):
    labels_flattened = labels.view(-1)
    string_labels = [INDEX_TO_LABEL[label.item()] if label.item() in INDEX_TO_LABEL else 'Unknown' for label in labels_flattened]
    
    for sample_idx in range(data.size(0)):
        if string_labels[sample_idx] != 'Unknown':  # Check if class label is not 'Unknown'
            print(f"Batch {batch_idx}: Point of Samples: {data[sample_idx, 1].cpu()}, Class Label: {string_labels[sample_idx]}")

    processed_samples.append((data, labels))
    
    if batch_idx == 2000:  
        break


# In[3]:


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


# In[ ]:


import torch
from torch.utils.data import DataLoader

def main():
    # Set the dataset directories
    lidar_root_dir = 'D:/data_object_velodyne'
    annotation_root_dir = 'D:/data_object_label_2'

    # Initialize the dataset
    kitti_dataset = KITTIDataset(lidar_root_dir, annotation_root_dir)

    # Create a DataLoader
    data_loader = DataLoader(kitti_dataset, batch_size=4, shuffle=True)

    # Define the model configuration
    num_classes = 2  # For example: Car, Cyclist
    dim_model = 512  # Embedding dimension
    num_heads = 8    # Number of heads in the multi-head attention mechanisms
    num_encoder_layers = 6  # Number of transformer encoder layers
    hidden_dim = 2048  # Dimension of the feedforward network in transformer
    num_proposals = 100  # Number of proposed objects

    # Initialize the transformer model
    model = LiDARTransformer(num_classes, dim_model, num_heads, num_encoder_layers, hidden_dim, num_proposals)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Define the loss functions for bounding box regression and classification
    criterion_bbox = torch.nn.SmoothL1Loss()
    criterion_class = torch.nn.CrossEntropyLoss()

    # Set up the optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Execute the training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0

        for batch_idx, (data, labels) in enumerate(data_loader):
            if data.size(0) == 0:
                # Skip the batch if it contains no data
                print(f"Skipping batch {batch_idx} due to empty data.")
                continue
            
            data = data.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            labels = labels.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            optimizer.zero_grad()

            # Perform a forward pass through the model
            bbox_preds, class_scores = model(data)

            # Calculate losses
            loss_bbox = criterion_bbox(bbox_preds, labels[:, :, :4])  # Assume bbox coordinates are in the first 4 columns
            loss_class = criterion_class(class_scores, labels[:, :, 4])  # Assume class indices start at the 5th column
            loss = loss_bbox + loss_class

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        scheduler.step()
        print(f"Epoch {epoch+1}, Total Loss: {total_loss/len(data_loader):.4f}")

        # Optionally save the model checkpoint
        if (epoch + 1) % 2 == 0:
            torch.save(model.state_dict(), f"lidar_transformer_epoch_{epoch+1}.pth")

    print("Training complete.")

if __name__ == "__main__":
    main()

