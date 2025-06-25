import time
import numpy as np
import math
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import *
from torch.autograd import Variable
import logging
import os
from datetime import datetime
from tool import EarlyStopping


# 设置日志目录
log_dir = "/home/wangjingbo/wjb/CCS2025/code/MVSTH/Milano/logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 获取当前时间并格式化为字符串
log_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# 设置日志文件名，包括时间戳
log_filename = os.path.join(log_dir, f"training_log_{log_timestamp}.txt")

# 设置日志配置
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,  # 设置日志级别
    format="%(asctime)s - %(levelname)s - %(message)s",  # 日志格式
)

# 创建日志记录器
logger = logging.getLogger()


def TrainMVSTH(train_dataloader, valid_dataloader, A, path, n_layer, d_model, knn_k, K=3, num_epochs=1, mamba_features=307, use_hypergraph=True, lr=1e-4):
    path = os.path.join("/home/wangjingbo/wjb/CCS2025/code/MVSTH/checkpoints/", path)
    if not os.path.exists(path):
        os.makedirs(path)
    # 'mamba_features=184' if we use Knowair dataset;
    # 'mamba_features=307' if we use PEMS04 datastet;
    # 'mamba_features=80' if we use HZ_Metro dataset; 
    # 'mamba_features=200' if we use milano_internet dataset;  or 400
    early_stopping = EarlyStopping(patience=10, verbose=True)

    inputs, labels = next(iter(train_dataloader))
    [batch_size, step_size, fea_size] = inputs.size()
    input_dim = fea_size
    hidden_dim = fea_size
    output_dim = fea_size

    mvsth_args = ModelArgs(
        K=K,
        A=torch.Tensor(A),
        feature_size=A.shape[0],
        d_model=d_model,  # hidden_dim is fea_size
        n_layer=n_layer,
        features=mamba_features,
        use_hypergraph=use_hypergraph,
        knn_k=knn_k
    )

    mvsth_mamba = MVSTH(mvsth_args)
    mvsth_mamba.cuda()
    
    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()

    learning_rate = lr
    optimizer = optim.AdamW(mvsth_mamba.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=False)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

    use_gpu = torch.cuda.is_available()

    interval = 10
    losses_train = []
    losses_interval_train = []
    losses_valid = []
    losses_interval_valid = []
    losses_epoch = []  # Initialize the list for epoch losses

    cur_time = time.time()
    pre_time = time.time()

    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.info('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        trained_number = 0

        valid_dataloader_iter = iter(valid_dataloader)

        valid_loss_list=[]
        for data in train_dataloader:
            inputs, labels = data

            if inputs.shape[0] != batch_size:
                continue

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            mvsth_mamba.zero_grad()

            labels = torch.squeeze(labels)
            pred = mvsth_mamba(inputs)  # Updated to use new model directly

            loss_train = loss_MSE(pred, labels)

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            # Update learning rate by CosineAnnealingLR
            scheduler.step()

            losses_train.append(loss_train.item())

            # validation
            try:
                # print("validation")
                inputs_val, labels_val = next(valid_dataloader_iter)
            except StopIteration:
                # print("StopIteration")
                valid_dataloader_iter = iter(valid_dataloader)
                inputs_val, labels_val = next(valid_dataloader_iter)

            if use_gpu:
                inputs_val, labels_val = Variable(inputs_val.cuda()), Variable(labels_val.cuda())
            else:
                inputs_val, labels_val = Variable(inputs_val), Variable(labels_val)

            labels_val = torch.squeeze(labels_val)

            with torch.no_grad():
                pred = mvsth_mamba(inputs_val)
                loss_valid = loss_MSE(pred, labels_val)
                losses_valid.append(loss_valid.item())
            
            # 存放每个epoch所有batchsize的vali loss
            valid_loss_list.append(loss_valid.item())

            trained_number += 1
            # print("trained_number:", trained_number)

            if trained_number % interval == 0:
                print("trained_number interval ++++++++++++++++++++++++:", trained_number)
                cur_time = time.time()
                loss_interval_train = np.around(sum(losses_train[-interval:]) / interval, decimals=8)
                losses_interval_train.append(loss_interval_train)
                loss_interval_valid = np.around(sum(losses_valid[-interval:]) / interval, decimals=8)
                losses_interval_valid.append(loss_interval_valid)
                logger.info('Iteration #: {}, train_loss: {}, valid_loss: {}, time: {}'.format(
                    trained_number * batch_size,
                    loss_interval_train,
                    loss_interval_valid,
                    np.around([cur_time - pre_time], decimals=8)))
                print('Iteration #: {}, train_loss: {}, valid_loss: {}, time: {}'.format(
                    trained_number * batch_size,
                    loss_interval_train,
                    loss_interval_valid,
                    np.around([cur_time - pre_time], decimals=8)))
                pre_time = cur_time

        epoch_vali_loss = np.around(sum(valid_loss_list) / len(valid_loss_list), decimals=8)
        early_stopping(epoch_vali_loss, mvsth_mamba, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return mvsth_mamba, [losses_train, losses_interval_train, losses_valid, losses_interval_valid]



def TestMVSTH(test_dataloader, A, path, max_speed, n_layer, d_model, knn_k, K=3, mamba_features=307, use_hypergraph=True):
    inputs, labels = next(iter(test_dataloader))
    [batch_size, step_size, fea_size] = inputs.size()

    path = os.path.join("/home/wangjingbo/wjb/CCS2025/code/MVSTH/checkpoints/", path)
    mvsth_args = ModelArgs(
            K=K,
            A=torch.Tensor(A),
            feature_size=A.shape[0],
            d_model=d_model,  # hidden_dim is fea_size
            n_layer=n_layer,
            features=mamba_features,
            use_hypergraph=use_hypergraph,
            knn_k=knn_k
        )

    mvsth_mamba = MVSTH(mvsth_args)
    mvsth_mamba.cuda()

    print('loading model')
    mvsth_mamba.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))

    mvsth_mamba.eval()

    cur_time = time.time()
    pre_time = time.time()

    use_gpu = torch.cuda.is_available()

    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()

    tested_batch = 0

    losses_mse = []
    losses_l1 = []
    MAEs = []
    MAPEs = []
    MSEs = []
    RMSEs = []
    VARs = []
    R_2s = []

    #predictions = []
    #ground_truths = []
    with torch.no_grad():
        for data in test_dataloader:
            inputs, labels = data

            if inputs.shape[0] != batch_size:
                continue

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            pred = mvsth_mamba(inputs)
            labels = torch.squeeze(labels)

            loss_mse = F.mse_loss(pred, labels)
            loss_l1 = F.l1_loss(pred, labels)
            MAE = torch.mean(torch.abs(pred - torch.squeeze(labels)))
            MAPE = torch.mean(torch.abs(pred - torch.squeeze(labels)) / torch.abs(torch.squeeze(labels)))
            # Calculate MAPE only for non-zero labels
            non_zero_labels = torch.abs(labels) > 0
            if torch.any(non_zero_labels):
                MAPE_values = torch.abs(pred - torch.squeeze(labels)) / torch.abs(torch.squeeze(labels))
                MAPE = torch.mean(MAPE_values[non_zero_labels])
                MAPEs.append(MAPE.item())

            MSE = torch.mean((torch.squeeze(labels) - pred)**2)
            RMSE = math.sqrt(torch.mean((torch.squeeze(labels) - pred)**2))
            VAR = 1-(torch.var(torch.squeeze(labels)-pred))/torch.var(torch.squeeze(labels))

            # 新增R平方计算（R²）
            SS_res = torch.sum((torch.squeeze(labels)-pred) ** 2)  # 残差平方和
            SS_tot = torch.sum(((torch.squeeze(labels)) - torch.mean((torch.squeeze(labels)))) ** 2)  # 总平方和
            R2 = 1 - (SS_res / SS_tot) if SS_tot != 0 else 0.0  # 处理总平方和为0的边界情况（所有label值相同）

            losses_mse.append(loss_mse.item())
            losses_l1.append(loss_l1.item())
            MAEs.append(MAE.item())
            MAPEs.append(MAPE.item())
            MSEs.append(MSE.item())
            RMSEs.append(RMSE)
            VARs.append(VAR.item())
            R_2s.append(R2.item())

            #predictions.append(pd.DataFrame(pred.cpu().data.numpy()))
            #ground_truths.append(pd.DataFrame(labels.cpu().data.numpy()))

            tested_batch += 1

            if tested_batch % 100 == 0:
                cur_time = time.time()
                logger.info('Tested #: {}, loss_l1: {}, loss_mse: {}, time: {}'.format(
                    tested_batch * batch_size,
                    np.around([loss_l1.data.cpu().numpy()], decimals=8),
                    np.around([loss_mse.data.cpu().numpy()], decimals=8),
                    np.around([cur_time - pre_time], decimals=8)))
                print('Tested #: {}, loss_l1: {}, loss_mse: {}, time: {}'.format(
                    tested_batch * batch_size,
                    np.around([loss_l1.data.cpu().numpy()], decimals=8),
                    np.around([loss_mse.data.cpu().numpy()], decimals=8),
                    np.around([cur_time - pre_time], decimals=8)))
                pre_time = cur_time

    losses_l1 = np.array(losses_l1)
    losses_mse = np.array(losses_mse)
    MAEs = np.array(MAEs)
    MAPEs = np.array(MAPEs)
    MSEs = np.array(MSEs)
    RMSEs = np.array(RMSEs)
    VARs = np.array(VARs)
    R_2s = np.array(R_2s)

    mean_l1 = np.mean(losses_l1) * max_speed
    std_l1 = np.std(losses_l1) * max_speed
    mean_mse = np.mean(losses_mse) * max_speed
    MAE_ = np.mean(MAEs) * max_speed
    std_MAE_ = np.std(MAEs) * max_speed #std_MAE measures the consistency & stability of the model's performance across different test sets or iterations. Usually if (std_MAE/MSE)<=10%., means the trained model is good.
    MAPE_ = np.mean(MAPEs) * 100
    MSE_ = np.mean(MSEs) * (max_speed ** 2)
    RMSE_ = np.mean(RMSEs) * max_speed
    VAR_ = np.mean(VARs)
    R_2_ = np.mean(R_2s)
    results = [MAE_, std_MAE_, MAPE_, MSE_, RMSE_, VAR_, R_2_]

    logger.info('Tested: MAE: {}, std_MAE: {}, MAPE: {}, MSE: {}, RMSE: {}, VAR: {}, R_2:{}'.format(MAE_, std_MAE_, MAPE_, MSE_, RMSE_, VAR_, R_2_))
    print('Tested: MAE: {}, std_MAE: {}, MAPE: {}, MSE: {}, RMSE: {}, VAR: {}, R_2:{}'.format(MAE_, std_MAE_, MAPE_, MSE_, RMSE_, VAR_, R_2_))

    return results