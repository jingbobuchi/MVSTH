export CUDA_VISIBLE_DEVICES=1
train_epoch=130
model_name='MVSTH'
data_name='milano_internet'
mamba_feature=400
batch_size=64
lr=0.0005
n_layer=4
hide_dim=400
knn_k=0.5

python -u main.py \
    -dataset $data_name \
    -model $model_name\
    -mamba_features $mamba_feature \
    -batch_size $batch_size \
    -lr $lr \
    -train_epoch $train_epoch \
    -n_layer $n_layer \
    -knn_k $knn_k
