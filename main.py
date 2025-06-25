import argparse
from prepare import *
from train_model import TrainMVSTH, TestMVSTH
import pandas as pd
import random

parser = argparse.ArgumentParser(description='Train & Test MVSTH for Network forecasting')

# choose dataset
parser.add_argument('-dataset', type=str, default='milano_internet', help='which dataset to run [options: know_air, pems04, hz_metro]')
# choose model
parser.add_argument('-model', type=str, default='MVSTH', help='which model to train & test [options: lstm]')
# choose number of node features.
parser.add_argument('-mamba_features', type=int, default=400, help='number of features for the MVSTH model [options: 307,184,80]')
# batch_size
parser.add_argument('-batch_size', type=int, default=48, help='number of features for the MVSTH model [options: 307,184,80]')
# lr
parser.add_argument('-lr', type=float, default=0.00001, help='number of features for the MVSTH model [options: 307,184,80]')
# train_epoch
parser.add_argument('-train_epoch', type=int, default=130, help='number of features for the MVSTH model [options: 307,184,80]')
# n_layer
parser.add_argument('-n_layer', type=int, default=4, help='number of n_layers')
# d_model
parser.add_argument('-d_model', type=int, default=400, help='d_model')
# knn_k
parser.add_argument('-knn_k', type=float, default=0.5, help='knn_k')

args = parser.parse_args()


# random seed
fix_seed = 3407
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(fix_seed)
    torch.cuda.manual_seed_all(fix_seed)

checkpoints_path = f"new_code_{args.dataset}_{args.model}_mf{args.mamba_features}_bs{args.batch_size}_lr{args.lr}_epoch{args.train_epoch}_nlayer{args.n_layer}_knnk{args.knn_k}"

###### loading data #######
if args.dataset =='milano_internet':
    print("\nLoading milano_internet data...")
    speed_matrix = pd.read_csv('/home/wangjingbo/wjb/CCS2025/code/MVSTH/datasets/Milano/milano_internet.csv',sep=',')
    A = np.load('/home/wangjingbo/wjb/CCS2025/code/MVSTH/datasets/Milano/adj_mat_milan.npy')

elif args.dataset == 'trentino_internet':
    print("\nLoading trentino_internet data...")
    speed_matrix = pd.read_csv('/home/wangjingbo/wjb/CCS2025/code/MVSTH/datasets/Trentino/trentino_internet.csv',sep=',')
    A = np.load('/home/wangjingbo/wjb/CCS2025/code/MVSTH/datasets/Trentino/adj_mat_trentino.npy')

print("\nPreparing train/test data...")
train_dataloader, valid_dataloader, test_dataloader, max_value = PrepareDataset(speed_matrix, BATCH_SIZE=args.batch_size)

# models you want to use
if args.model == 'MVSTH':
    print("\nTraining MVSTH model...")
    MVSTHmamba, MVSTH_loss = TrainMVSTH(train_dataloader, valid_dataloader, A, path=checkpoints_path, n_layer=args.n_layer, d_model=args.d_model, knn_k=args.knn_k, K=3, num_epochs=args.train_epoch, mamba_features=args.mamba_features, use_hypergraph=True, lr=args.lr)
    print("\nTesting MVSTH model...")
    results = TestMVSTH(test_dataloader, A=A, path=checkpoints_path, max_speed=max_value, n_layer=args.n_layer, d_model=args.d_model, knn_k=args.knn_k, K=3, mamba_features=args.mamba_features, use_hypergraph=True)