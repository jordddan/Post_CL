gpu_device=$1
data=$2
src_data_path=/opt/data/private/noise_master/TC_dataset/${data}/train.csv

src_lan=$3
tar_lan=$4

tar_data_path=/opt/data/private/noise_master/TC_dataset/${data}/train_aug_${src_lan}_${tar_lan}.csv

CUDA_VISIBLE_DEVICES=$gpu_device python /opt/data/private/noise_master/data_utils/back_trans.py \
    --src_data_path $src_data_path \
    --tar_data_path $tar_data_path \
    --src_lan $src_lan \
    --tar_lan $tar_lan