import os

# 【第一步】必须在所有科学计算库导入之前设置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 如果还是报错，可以加上下面这行强制单线程
os.environ["OMP_NUM_THREADS"] = "1"

import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split



class BraTSDataset(Dataset):
    def __init__(self, root_dir='./data', transform=None, split='all', test_size=0.3, random_state=42):
        self.root_dir = root_dir
        self.transform = transform
        # 只保留文件夹名称，过滤掉可能存在的杂文件
        all_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
        
        if split == 'all':
            self.folder_list = all_folders
        else:
            train_folders, test_folders = train_test_split(all_folders, test_size=test_size, random_state=random_state)
            if split == 'train':
                self.folder_list = train_folders
            elif split == 'test':
                self.folder_list = test_folders
            else:
                raise ValueError("split must be 'train', 'test', or 'all'")

    def __len__(self):
        return len(self.folder_list)

    def __getitem__(self, idx):
        # 1. 获取当前样本的文件夹路径
        folder_name = self.folder_list[idx]
        folder_path = os.path.join(self.root_dir, folder_name)

        # 2. 定义四种模态的文件名后缀
        modalities = ['flair', 't1', 't1ce', 't2']
        
        # 3. 加载并堆叠 4 个模态的数据 (C, H, W, D)
        images = []
        for mod in modalities:
            file_path = os.path.join(folder_path, f"{folder_name}_{mod}.nii.gz")
            img = nib.load(file_path).get_fdata()
            images.append(img)
        
        # 将 list 转为 numpy 再转为 tensor，形状变为 (4, H, W, D)
        image_stack = np.stack(images, axis=0).astype(np.float32)

        # 4. 加载标签 (Segmentation)
        label_path = os.path.join(folder_path, f"{folder_name}_seg.nii.gz")
        label = nib.load(label_path).get_fdata().astype(np.float32)
        label[label == 4] = 3  # 将标签4合并到3，确保标签值为0,1,2,3


        # 5. 应用变换 (Transform)
        if self.transform:
            image_stack, label = self.transform(image_stack, label)

        return torch.from_numpy(image_stack), torch.from_numpy(label)


class Dataloader:
    def __init__(self, dataset, batch_size=10, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def get_loader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)



# 总数据集大小: 1251
# 训练集大小: 875 (69.94%)
# 测试集大小: 376 (30.06%)
# 测试加载训练数据的一个batch...
# 训练Batch 图像形状: torch.Size([4, 4, 240, 240, 155])
# 训练Batch 标签形状: torch.Size([4, 240, 240, 155])
# 测试加载测试数据的一个batch...
# 测试Batch 图像形状: torch.Size([4, 4, 240, 240, 155])
# 测试Batch 标签形状: torch.Size([4, 240, 240, 155])

