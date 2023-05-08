epoch=8
gpu_device=$1
pretrain_model_dir=data/pre_train_models/


# dataset=sst2
# data=${dataset}_asym_0.4
# train_path=/opt/data/private/qd/noise_master/TC_dataset/$dataset/train_asym_0.4.csv
# test_path=/opt/data/private/qd/noise_master/TC_dataset/$dataset/test.csv


dataset=imdb
data=${dataset}_asym_0.4
train_path=/opt/data/private/qd/noise_master/TC_dataset/$dataset/train_10000_asym_0.4.csv
test_path=/opt/data/private/qd/noise_master/TC_dataset/$dataset/test.csv


# dataset=amazon
# train_path=/opt/data/private/qd/noise_master/TC_dataset/$dataset/train_20000.csv
# test_path=/opt/data/private/qd/noise_master/TC_dataset/$dataset/test_20000.csv

# dataset=yahoo
# data=yahoo
# train_path=/opt/data/private/qd/noise_master/TC_dataset/$dataset/train_20000.csv
# test_path=/opt/data/private/qd/noise_master/TC_dataset/$dataset/test_20000.csv


# dataset=dbpedia
# train_path=/opt/data/private/qd/noise_master/TC_dataset/$dataset/train_20000.csv
# test_path=/opt/data/private/qd/noise_master/TC_dataset/$dataset/test_20000.csv

# dataset=yahoo
# data=${dataset}_asym_0.4
# train_path=/opt/data/private/qd/noise_master/TC_dataset/$dataset/train_20000_asym_0.4.csv
# test_path=/opt/data/private/qd/noise_master/TC_dataset/$dataset/test_20000.csv


# dataset=amazon
# data=${dataset}
# train_path=/opt/data/private/qd/noise_master/TC_dataset/$dataset/train.csv
# test_path=/opt/data/private/qd/noise_master/TC_dataset/$dataset/test.csv

pre_trained_data=$2

model=prajjwal1/bert-small

noise_type=asym
lr=1e-5
ALPHA=(0)
SEED=(1 2 8)
BETA=(0)
cl=simcse
MLM_W=(0)
for alpha in ${ALPHA[@]};do
    for seed in ${SEED[@]};do
        for beta in ${BETA[@]};do
            for mlm_w in ${MLM_W[@]};do
        log_path=/opt/data/private/qd/noise_master/post_training/log/post_training/$data/$pre_trained_data
        rm -rf log_path
        mkdir -p $log_path
            CUDA_VISIBLE_DEVICES=$gpu_device python train_arch1.py \
                --bert_type $model --pre_trained_data $pre_trained_data \
                --train_path $train_path \
                --test_path $test_path \
                --log_path $log_path/${seed}.txt \
                --seed $seed \
                --batch_size 32 \
                --learning_rate $lr \
                --alpha $alpha --beta $beta --cl $cl --mlm_w $mlm_w \
                --epoch $epoch \
                --show_bar
            done
        done
    done
done