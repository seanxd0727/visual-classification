import timeit
import warnings
from datetime import datetime
import os
from scipy.stats import pearsonr
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable
from dataset import Dataset_pain
from network import C3D_model, R2Plus1D_model, R3D_model

warnings.filterwarnings("ignore")  # 抑制警告
# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

snapshot = 50  # 在每个snapshot存储一个模型
lr = 1e-3  # 学习率 c3d=1e-4 r21d=1e-3 R3D=1e-3

txt_name = 'rawframe'
num_classes = 6
num_epoch = 50  # 训练总次数
save_dir = os.path.join("run")
modelName = 'R3D'  # Options: C3D or R2Plus1D or R3D
saveName = modelName + '-' + txt_name
load_dir = os.path.join("run", "models", modelName + "-rawframe.pth")

def train_model(txt_name=txt_name, save_dir=save_dir, num_classes=num_classes, lr=lr, nEpochs=num_epoch,
                save_epoch=snapshot):
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

    # model.load_state_dict(torch.load(load_dir))  # 载入模型

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

    # 创建日志目录
    log_dir = os.path.join(save_dir, 'data', modelName, datetime.now().strftime('%b%d_%H-%M-%S'))
    writer = SummaryWriter(log_dir=log_dir)

    print('Training model on {} .txt'.format(txt_name))

    data = Dataset_pain(txt_name=txt_name, slip_len=16, step=5)
    train_dataset, val_dataset = random_split(dataset=data, lengths=[0.8, 0.2])  # 按8：2分割数据集
    train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=5, num_workers=4)
    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}  # 对应数据集大小

    for epoch in range(0, nEpochs):
        for phase in ['train', 'val']:

            epoch_preds = []
            epoch_labels = []

            # 获取当前时间戳，以便后续计算时间消耗。
            start_time = timeit.default_timer()

            # 将变量重置为0
            # 以便开始新的训练或验证周期时重新计算损失和正确率
            running_loss = 0.0
            running_corrects = 0.0
            running_mae_loss = 0.0
            running_mse_loss = 0.0

            if phase == 'train':
                # 更新学习率，并将模型设置为训练模式
                scheduler.step()
                model.train()
            else:
                # 将模型设置为评估模式
                model.eval()

            for inputs, labels in trainval_loaders[phase]:
                # 对于每个批次的数据(inputs和labels),将它们移动到正在进行训练的设备上(即GPU或CPU)
                inputs = Variable(inputs, requires_grad=True).to(device)
                labels = Variable(labels).to(device)
                optimizer.zero_grad()  # 将优化器的状态重置为0

                if phase == 'train':
                    outputs = model(inputs)
                else:
                    with torch.no_grad():
                        outputs = model(inputs)  # 禁用梯度计算

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

                if phase == 'train':
                    loss.backward()  # 反向传播
                    optimizer.step()  # 更新优化器

                running_loss += loss.item() * inputs.size(0)  # 总损失值
                running_corrects += torch.sum(preds == labels.data)  # 计算预测值(preds)和真实标签(labels)之间的差异
                running_mae_loss += mae_loss.item()
                running_mse_loss += mse_loss.item()

            epoch_loss = running_loss / trainval_sizes[phase]  # 每个周期的平均损失值
            epoch_acc = running_corrects.double() / trainval_sizes[phase]  # 每个周期的平均准确率
            epoch_mae = running_mae_loss / trainval_sizes[phase]  # mae
            epoch_mse = running_mse_loss / trainval_sizes[phase]  # mse
            if epoch_preds[1:] == epoch_preds[:-1] or epoch_labels[1:] == epoch_labels[:-1]:
                epoch_pcc = 0.0
            else:
                epoch_pcc = abs(pearsonr(epoch_preds, epoch_labels)[0])

            # 记录日志,便于形成图表（tensorboard）
            if phase == 'train':
                writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
                writer.add_scalar('data/train_mae_epoch', epoch_mae, epoch)
                writer.add_scalar('data/train_mse_epoch', epoch_mse, epoch)
                writer.add_scalar('data/train_pcc_epoch', epoch_pcc, epoch)
            else:
                writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch)
                writer.add_scalar('data/val_mae_epoch', epoch_mae, epoch)
                writer.add_scalar('data/val_mse_epoch', epoch_mse, epoch)
                writer.add_scalar('data/val_pcc_epoch', epoch_pcc, epoch)

            print("[{}] Epoch: {}/{} CrossEntropyLoss: {} Acc: {} MAE: {} MSE: {} PCC: {}".format(phase, epoch + 1,
                                                                                                  nEpochs, epoch_loss,
                                                                                                  epoch_acc, epoch_mae,
                                                                                                  epoch_mse, epoch_pcc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

        # 保存模型
        if (epoch+1) % save_epoch == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, 'models', saveName + '.pth'))
            print("Save model at {}\n".format(os.path.join(save_dir, 'models', saveName + '.pth')))
    writer.close()


if __name__ == "__main__":
    train_model()
