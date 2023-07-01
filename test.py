import timeit
import warnings
import os
from scipy.stats import pearsonr
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from dataset import Dataset_pain
from network import C3D_model, R2Plus1D_model, R3D_model

warnings.filterwarnings("ignore")  # 抑制警告

num_classes = 6
modelName = "C3D"  # 模型名
load_dir = os.path.join("run", "models", modelName + "-rawframe.pth")
people_id = ["ll042", "jh043", "jl047", "aa048", "bm049", "dr052", "fn059", "ak064", "mg066", "bn080", "ch092",
             "tv095", "bg096", "gf097", "mg101", "jk103", "nm106", "hs107", "th108", "ib109", "jy115", "kz120",
             "vw121", "jh123", "dn124"]


def test_model(load_dir=load_dir, num_classes=num_classes):
    # 载入模型C3D or R2Plus1D or R3D
    if modelName == 'C3D':
        model = C3D_model.C3D(num_classes=num_classes, pretrained=True)
    elif modelName == 'R2Plus1D':
        model = R2Plus1D_model.R2Plus1DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
    elif modelName == 'R3D':
        model = R3D_model.R3DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
    else:
        print("没有在这个模型！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！")

    model.load_state_dict(torch.load(load_dir))
    model.eval()
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失(多分类)
    MAE_loss = nn.L1Loss()  # MAE
    MSE_loss = nn.MSELoss()  # MSE

    for person_id in people_id:
        data = Dataset_pain(txt_name=person_id, slip_len=16, step=5)
        test_dataloader = DataLoader(data, batch_size=5, shuffle=True, num_workers=4)
        test_sizes = len(data)

        epoch_preds = []
        epoch_labels = []

        # 获取当前时间戳，以便后续计算时间消耗。
        start_time = timeit.default_timer()

        # 将变量重置为0
        # 以便开始新周期时重新计算损失和正确率
        running_loss = 0.0
        running_corrects = 0.0
        running_mae_loss = 0.0
        running_mse_loss = 0.0

        for inputs, labels in test_dataloader:
            outputs = model(inputs)
            # 对输出(outputs)进行softmax归一化
            probs = nn.Softmax(dim=1)(outputs)
            # 计算预测值
            preds = torch.max(probs, 1)[1]
            # 计算当前批次的损失值,loss=CrossEntropyLoss, 交叉熵损失
            loss = criterion(outputs, labels)
            mae_loss = MAE_loss(preds, labels.type(torch.float64))
            mse_loss = MSE_loss(preds, labels.type(torch.float64))

            # 保存预测值和标签以计算pcc
            epoch_preds.extend(preds.tolist())
            epoch_labels.extend(labels.tolist())

            running_loss += loss.item() * inputs.size(0)  # 总损失值
            running_corrects += torch.sum(preds == labels.data)  # 计算预测值(preds)和真实标签(labels)之间的差异
            running_mae_loss += mae_loss.item()
            running_mse_loss += mse_loss.item()

        epoch_loss = running_loss / test_sizes  # 每个周期的平均损失值
        epoch_acc = running_corrects.double() / test_sizes  # 每个周期的平均准确率
        epoch_mae = running_mae_loss / test_sizes  # mae
        epoch_mse = running_mse_loss / test_sizes  # mse
        if epoch_preds[1:] == epoch_preds[:-1] or epoch_labels[1:] == epoch_labels[:-1]:
            epoch_pcc = 0.0
        else:
            epoch_pcc = abs(pearsonr(epoch_preds, epoch_labels)[0])

        print("[test] person_id: {} CrossEntropyLoss: {} Acc: {} MAE: {} MSE: {} PCC: {}".format(person_id, epoch_loss,
                                                                                                 epoch_acc, epoch_mae,
                                                                                                 epoch_mse, epoch_pcc))
        stop_time = timeit.default_timer()
        print("Execution time: " + str(stop_time - start_time) + "\n")


if __name__ == "__main__":
    test_model()
