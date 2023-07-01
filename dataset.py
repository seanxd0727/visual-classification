import os
import cv2
import numpy as np
import torch
from torch.utils import data


class Dataset_pain(data.Dataset):

    def __init__(self, txt_name, slip_len, step):
        self.txt_name = txt_name
        self.slip_len = slip_len
        self.step = step
        self.height = 112
        self.width = 112
        self.data_txt = []
        self.sum_len = (slip_len - 1) * step + 1
        txt_path = os.path.join("data_txt", txt_name + ".txt")
        with open(txt_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                check_line = line.split()
                if int(check_line[1]) > self.sum_len:
                    self.data_txt.append(line)

    def __len__(self):
        """返回样本总数"""
        return len(self.data_txt)

    def __getitem__(self, index):
        """生成一个数据样本"""
        line_data = self.data_txt[index]
        line_data = line_data.strip()
        line_data = line_data.split()
        address = line_data[0]
        num_frames = line_data[1]
        label = np.array(int(line_data[2]))
        buffer = self.load_frames(address)
        buffer = self.crop(buffer)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(label)

    def load_frames(self, file_dir):
        frames = sorted(os.listdir(file_dir))
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.height, self.width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(os.path.join(file_dir, frame_name))).astype(np.float64)
            frame = np.resize(frame, [self.height, self.width, 3])
            buffer[i] = frame
        return buffer

    def crop(self, buffer):
        time_index = np.random.randint(buffer.shape[0] - self.sum_len)
        buffer = buffer[time_index:time_index + self.sum_len:self.step, :, :, :]
        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            # frame = (frame - frame.min(axis=0)) / (frame.max(axis=0) - frame.min(axis=0))
            buffer[i] = frame
        return buffer
