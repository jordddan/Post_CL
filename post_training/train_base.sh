epoch=$1
gpu_device=4
pretrain_model_dir=data/pre_train_models/


# data=trec
# train_path=data/trec/train_n.csv
# test_path=data/trec/test_n.csv

# data=imdb45k
# train_path=data/IMDB/train.csv
# test_path=data/IMDB/test.csv

# data=trec
# data=sst2
# train_path=/opt/data/private/qd/noise_master/TC_dataset/$data/train_asym_0.4.csv
# test_path=/opt/data/private/qd/noise_master/TC_dataset/$data/test.csv


# data=amazon
# train_path=/opt/data/private/qd/noise_master/TC_dataset/$data/train_20000.csv
# test_path=/opt/data/private/qd/noise_master/TC_dataset/$data/test.csv

# dataset=imdb
# data=imdb_aysm_0.4
# train_path=/opt/data/private/qd/noise_master/TC_dataset/$dataset/train_10000_asym_0.4.csv
# test_path=/opt/data/private/qd/noise_master/TC_dataset/$dataset/test.csv

# dataset=agnews
# data=agnews_aysm_0.4
# train_path=/opt/data/private/qd/noise_master/TC_dataset/$dataset/train_10000_asym_0.4.csv
# test_path=/opt/data/private/qd/noise_master/TC_dataset/$dataset/test.csv


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

dataset=yahoo
data=${dataset}_asym_0.4
train_path=/opt/data/private/qd/noise_master/TC_dataset/$dataset/train_20000_asym_0.4.csv
test_path=/opt/data/private/qd/noise_master/TC_dataset/$dataset/test_20000.csv


model=prajjwal1/bert-small

noise_type=asym
lr=1e-5
arr=(0.1)
arr2=(1 2 8)
for alpha in ${arr[@]};do
    for seed in ${arr2[@]};do

    log_path=/opt/data/private/qd/noise_master/baselines/log/base/$data/
    rm -rf log_path
    mkdir -p $log_path
        CUDA_VISIBLE_DEVICES=$gpu_device python train_base.py \
            --bert_type $model \
            --train_path $train_path \
            --test_path $test_path \
            --log_path $log_path/${seed}.txt \
            --seed $seed \
            --noise_type $noise_type \
            --batch_size 32 \
            --learning_rate $lr \
            --epoch $epoch \
            --show_bar 
    done
done