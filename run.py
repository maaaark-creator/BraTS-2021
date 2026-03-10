import os

# 【第一步】必须在所有科学计算库导入之前设置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 如果还是报错，可以加上下面这行强制单线程
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim
from data_preprocessing import BraTSDataset
from model import UNet3D


if __name__ == '__main__':

    train_dataset = BraTSDataset(root_dir='./data', split='train', test_size=0.3, random_state=42)
    test_dataset = BraTSDataset(root_dir='./data', split='test', test_size=0.3, random_state=42)

    print(f"总数据集大小: {len(train_dataset.folder_list) + len(test_dataset.folder_list)}")
    print(f"训练集大小: {len(train_dataset)} ({len(train_dataset) / (len(train_dataset) + len(test_dataset)):.2%})")
    print(f"测试集大小: {len(test_dataset)} ({len(test_dataset) / (len(train_dataset) + len(test_dataset)):.2%})")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)  
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet3D(in_channels=4, out_channels=4).to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)


    num_epochs = 1  
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device).long()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")


    # 6. 测试模型
    model.eval()
    test_loss = 0.0
    dice_scores = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).long()
            outputs = model(images)
            

            loss = criterion(outputs, labels)
            test_loss += loss.item()
            

            predictions = torch.argmax(outputs, dim=1)  # (batch_size, H, W, D)
            

            for class_idx in range(1, 4):  # 跳过背景类别0
                pred_mask = (predictions == class_idx).float()
                true_mask = (labels == class_idx).float()
                
                intersection = torch.sum(pred_mask * true_mask)
                union = torch.sum(pred_mask) + torch.sum(true_mask)
                
                if union > 0:
                    dice = (2.0 * intersection) / union
                    dice_scores.append(dice.item())
    
    avg_test_loss = test_loss / len(test_loader)
    avg_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0.0
    
    print(f"测试完成, 平均测试损失: {avg_test_loss:.4f}")
    print(f"平均Dice系数: {avg_dice:.4f}")
    
    print("训练完成！")
