import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math

LABEL_TO_INDEX = {'car': 0, 'bicycle': 1, 'pedestrian': 2}  # Adjusted for your labels
INDEX_TO_LABEL = {0: 'Car', 1: 'Bicycle', 2: 'Pedestrian'}  # Adjusted for your labels

class NuScenesDataset(Dataset):
    def __init__(self, root_dir, annotation_file, max_points=50000, num_files=10000):  # Reduced max_points to 50000
        self.root_dir = root_dir
        self.annotation_file = annotation_file
        self.max_points = max_points
        self.num_files = num_files
        self.lidar_files = self.get_lidar_files()[:self.num_files]  # Slice to get only the first num_files
        self.annotations = self.load_annotations()

        # Debug print statements
        print(f"Found {len(self.lidar_files)} LiDAR files. Processing {self.num_files} files.")
        print(f"Loaded {len(self.annotations)} annotations.")

    def get_lidar_files(self):
        files = []
        split_dir = os.path.join(self.root_dir, 'v1.0-trainval')
        print(f"Looking for LiDAR files in {split_dir}")
        
        if not os.path.exists(split_dir):
            print(f"Directory {split_dir} does not exist.")
            return files
        
        for root, _, filenames in os.walk(split_dir):
            for filename in filenames:
                if filename.endswith('.bin'):
                    full_path = os.path.join(root, filename)
                    files.append(full_path)
        
        return files

    def load_annotations(self):
        with open(self.annotation_file, 'r') as file:
            data = json.load(file)
        return data

    def __len__(self):
        return len(self.lidar_files)

    def __getitem__(self, idx):
        lidar_file = self.lidar_files[idx]
        lidar_data = np.fromfile(lidar_file, dtype=np.float32)
        
        if lidar_data.size % 4 != 0:
            raise ValueError(f"LiDAR data from {lidar_file} cannot be reshaped to (-1, 4). Incorrect number of elements.")
        
        lidar_data = lidar_data.reshape(-1, 4)
        
        # Directly pad or truncate LiDAR data as necessary
        padded_lidar_data = self.pad_or_truncate(lidar_data)
        
        # Get sample token based on the file name
        sample_token = os.path.basename(lidar_file).split('.')[0]
        annotations = self.get_annotations(sample_token)
        
        return torch.tensor(padded_lidar_data, dtype=torch.float32), annotations

    def pad_or_truncate(self, lidar_data):
        num_points = lidar_data.shape[0]
        if num_points < self.max_points:
            padding = np.zeros((self.max_points - num_points, lidar_data.shape[1]), dtype=np.float32)
            lidar_data = np.vstack((lidar_data, padding))
        elif num_points > self.max_points:
            lidar_data = lidar_data[:self.max_points, :]
        return lidar_data

    def get_annotations(self, sample_token):
        # Retrieve annotations for the given sample token
        for item in self.annotations:
            if item['token'] == sample_token:
                return item['annotations']
        return []

def process_objects_in_lidar(data, annotations):
    detected_objects = []
    for annotation in annotations:
        class_label = LABEL_TO_INDEX.get(annotation['label_name'], -1)
        if class_label != -1:
            bbox_center = np.array(annotation['translation'])
            bbox_size = np.array(annotation['size'])
            bbox_min = bbox_center - bbox_size / 2
            bbox_max = bbox_center + bbox_size / 2
            
            in_bbox_indices = np.where(
                (data[:, 0] >= bbox_min[0]) & (data[:, 0] <= bbox_max[0]) &
                (data[:, 1] >= bbox_min[1]) & (data[:, 1] <= bbox_max[1]) &
                (data[:, 2] >= bbox_min[2]) & (data[:, 2] <= bbox_max[2])
            )[0]
            
            for idx in in_bbox_indices:
                detected_objects.append((data[idx], class_label))
    
    return detected_objects

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
            # Ensure the mask has the same shape as the attention logits
            mask = mask.unsqueeze(1).unsqueeze(2)
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

# Training loop and loss function
def train_model(model, dataloader, optimizer, criterion, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, annotations) in enumerate(dataloader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            bbox_preds, class_scores = model(inputs)
            
            # Debug print statements to check tensor sizes
            print(f"bbox_preds shape: {bbox_preds.shape}")
            print(f"class_scores shape: {class_scores.shape}")
            print(f"annotations length: {len(annotations)}")

            # Compute loss
            loss = criterion(bbox_preds, class_scores, annotations)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100}")
                running_loss = 0.0

    # Save the model weights
    torch.save(model.state_dict(), 'Mo-Li3DeSTr-R-New.pth')

# Placeholder Loss Function
def detection_loss(bbox_preds, class_scores, annotations):
    # This function should compute the loss based on predicted bounding boxes and class scores vs annotations
    # Placeholder implementation, actual implementation should compute the actual loss
    return torch.tensor(0.0, requires_grad=True)

# Initialize model, optimizer, and DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LiDARTransformer(num_classes=2, dim_model=128, num_heads=4, num_encoder_layers=4, hidden_dim=256, num_proposals=100).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = detection_loss

root_dir = r"D:\nuScenes-lidarseg"
annotation_file = r"D:\nuScenes_Labels\lidarseg.json"
nuscenes_dataset = NuScenesDataset(root_dir, annotation_file, num_files=10000)
data_loader = DataLoader(nuscenes_dataset, batch_size=2, shuffle=True)  # Reduced batch size to 2

# Train the model
train_model(model, data_loader, optimizer, criterion, num_epochs=10)
