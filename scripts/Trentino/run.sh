export CUDA_VISIBLE_DEVICES=0
train_epoch=130
model_name='MVSTH'
data_name='trentino_internet'
mamba_features=400
batch_size=16
lr=0.0001
n_layer=4
d_model=400
knn_k=0.5

python -u main.py \
    -dataset $data_name \
    -model $model_name\
    -mamba_features $mamba_features \
    -batch_size $batch_size \
    -lr $lr \
    -train_epoch $train_epoch \
    -n_layer $n_layer \
    -d_model $d_model \
    -knn_k $knn_k