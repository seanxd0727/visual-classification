import timeit
import warnings
from scipy.stats import pearsonr
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import Dataset_pain
from network import C3D_model, R2Plus1D_model, R3D_model

warnings.filterwarnings("ignore")  # 抑制警告
# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

lr = 1e-3  # 学习率 c3d=1e-4 r21d=1e-3 R3D=1e-3
people_id = ["ll042", "jh043", "jl047", "aa048", "bm049", "dr052", "fn059", "ak064", "mg066", "bn080", "ch092",
             "tv095", "bg096", "gf097", "mg101", "jk103", "nm106", "hs107", "th108", "ib109", "jy115", "kz120",
             "vw121", "jh123", "dn124"]
num_classes = 6
modelName = 'R3D'  # Options: C3D or R2Plus1D or R3D


def train_model():
    # 载入模型C3D or R2Plus1D or R3D
    if modelName == 'C3D':
        model = C3D_model.C3D(num_classes=num_classes, pretrained=True)
        train_params = [{'params': C3D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': C3D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R2Plus1D':
        model = R2Plus1D_model.R2Plus1DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        train_params = [{'params': R2Plus1D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': R2Plus1D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R3D':
        model = R3D_model.R3DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        train_params = [{'params': R3D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': R3D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    else:
        print("没有在这个模型！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！")

    criterion = nn.CrossEntropyLoss()  # 交叉熵损失(多分类)
    MAE_loss = nn.L1Loss()  # MAE
    MSE_loss = nn.MSELoss()  # MSE
    # 用于训练模型，其中 lr 参数设置初始学习率，momentum 参数设置动量因子，weight_decay 参数设置 L2 正则化系数
    optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    # 用于在训练过程中调整学习率，其中 step_size 参数设置每个调整之间的周期数，gamma 参数设置每次调整后学习率降低的因子
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 打印模型的总参数数量，并将模型和损失函数移动到指定的设备(例如GPU)上
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)
    criterion.to(device)

    print("进行25折交叉验证")

    torch.save(model.state_dict(), 'model_begin_weight.pth')  # 保存模型初始化权重

    for person_id in people_id:
        start_time = timeit.default_timer()  # 获取当前时间戳，以便后续计算时间消耗。
        model.load_state_dict(torch.load('model_begin_weight.pth'))  # 加载权重使每次开始时权重相同，即初始化模型

        # 得到其他人的id
        others_id = ["ll042", "jh043", "jl047", "aa048", "bm049", "dr052", "fn059", "ak064", "mg066", "bn080", "ch092",
                     "tv095", "bg096", "gf097", "mg101", "jk103", "nm106", "hs107", "th108", "ib109", "jy115", "kz120",
                     "vw121", "jh123", "dn124"]
        others_id.remove(person_id)

        val_dataset = Dataset_pain(txt_name=person_id, slip_len=16, step=5)
        val_dataloader = DataLoader(val_dataset, batch_size=5, num_workers=4)

        for phase in ['train', 'val']:
            epoch_preds = []
            epoch_labels = []

            # 将变量重置为0
            # 以便开始新的训练或验证周期时重新计算损失和正确率
            running_loss = 0.0
            running_corrects = 0.0
            running_mae_loss = 0.0
            running_mse_loss = 0.0
            data_size = 0

            if phase == 'train':
                # 更新学习率，并将模型设置为训练模式
                scheduler.step()
                model.train()

                # 对其他人依次训练
                for other_id in others_id:
                    train_dataset = Dataset_pain(txt_name=other_id, slip_len=16, step=5)
                    train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=4)
                    data_size += len(train_dataset)
                    for inputs, labels in train_dataloader:
                        # 对于每个批次的数据(inputs和labels),将它们移动到正在进行训练的设备上(即GPU或CPU)
                        inputs = Variable(inputs, requires_grad=True).to(device)
                        labels = Variable(labels).to(device)
                        outputs = model(inputs)
                        optimizer.zero_grad()  # 将优化器的状态重置为0

                        # 计算当前批次的损失值,loss=CrossEntropyLoss, 交叉熵损失
                        loss = criterion(outputs, labels)
                        loss.backward()  # 反向传播
                        optimizer.step()  # 更新优化器
            else:
                model.eval()  # 将模型设置为评估模式
                data_size = len(val_dataset)
                for inputs, labels in val_dataloader:
                    # 对于每个批次的数据(inputs和labels),将它们移动到正在进行训练的设备上(即GPU或CPU)
                    inputs = Variable(inputs, requires_grad=True).to(device)
                    labels = Variable(labels).to(device)
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

                epoch_loss = running_loss / data_size  # 每个周期的平均损失值
                epoch_acc = running_corrects.double() / data_size  # 每个周期的平均准确率
                epoch_mae = running_mae_loss / data_size  # mae
                epoch_mse = running_mse_loss / data_size  # mse
                if epoch_preds[1:] == epoch_preds[:-1] or epoch_labels[1:] == epoch_labels[:-1]:
                    epoch_pcc = 0.0  # 无法计算pcc
                else:
                    epoch_pcc = abs(pearsonr(epoch_preds, epoch_labels)[0])

                print("[{}] person_id: {} CrossEntropyLoss: {} Acc: {} MAE: {} MSE: {} PCC: {}".format(phase,
                                                                                                       person_id,
                                                                                                       epoch_loss,
                                                                                                       epoch_acc,
                                                                                                       epoch_mae,
                                                                                                       epoch_mse,
                                                                                                       epoch_pcc))
                stop_time = timeit.default_timer()
                print("Execution time: " + str(stop_time - start_time) + "\n")


if __name__ == "__main__":
    train_model()
