epoch=6
gpu_device=$1
pretrain_model_dir=data/pre_train_models/



# dataset=sst2
# train_path=/opt/data/private/noise_master/TC_dataset/$dataset/train_merged.csv
# test_path=/opt/data/private/noise_master/TC_dataset/$dataset/test.csv
##train_aug1=/opt/data/private/noise_master/TC_dataset_ss/$dataset/train_aug_en_de.csv
##train_aug2=/opt/data/private/noise_master/TC_dataset_ss/$dataset/train_aug_en_fr.csv

# dataset=imdb
# train_path=/opt/data/private/noise_master/TC_dataset/imdb/train_20000.csv
# test_path=/opt/data/private/noise_master/TC_dataset/imdb/test.csv

dataset=amazon
train_path=/opt/data/private/noise_master/TC_dataset/$dataset/train_20000.csv
test_path=/opt/data/private/noise_master/TC_dataset/$dataset/test_20000.csv

# dataset=yahoo
# train_path=/opt/data/private/noise_master/TC_dataset/$dataset/train_20000.csv
# test_path=/opt/data/private/noise_master/TC_dataset/$dataset/test_20000.csv


# dataset=agnews
# train_path=/opt/data/private/noise_master/TC_dataset/$dataset/train_20000.csv
# test_path=/opt/data/private/noise_master/TC_dataset/$dataset/test.csv

# dataset=dbpedia
# train_path=/opt/data/private/noise_master/TC_dataset/$dataset/train_20000.csv
# test_path=/opt/data/private/noise_master/TC_dataset/$dataset/test_20000.csv




pre_trained_data=$2

model=prajjwal1/bert-small

noise_type=asym
noise_ratio=0.4
num_k=500

lr=1e-5
# CL learning
ALPHA=(0.1)
BETA=(0.1)
warmup_epoch=2

# base
# ALPHA=(0)
# BETA=(0)
# warmup_epoch=7

SEED=(1 2 8)
cl=simcse
MLM_W=(0)
for alpha in ${ALPHA[@]};do
    for seed in ${SEED[@]};do
        for beta in ${BETA[@]};do
            for mlm_w in ${MLM_W[@]};do
        log_path=/opt/data/private/noise_master/rep_ss/log/$dataset/${pre_trained_data}_noisetype_${noise_type}_noiseratio_${noise_ratio}
        rm -rf log_path
        mkdir -p $log_path
            CUDA_VISIBLE_DEVICES=$gpu_device python train_arch1.py \
                --bert_type $model --seed $seed --pre_trained_data $pre_trained_data \
                --noise_type $noise_type --noise_ratio $noise_ratio \
                --train_path $train_path --test_path $test_path  \
                --log_path $log_path/${seed}.txt \
                --batch_size 32 --learning_rate $lr  --epoch $epoch \
                --alpha $alpha --beta $beta --cl $cl --mlm_w $mlm_w --num_k $num_k --warmup_epoch $warmup_epoch \
                --show_bar 
            done
        done
    done
done

noise_type=asym
noise_ratio=0.2

for alpha in ${ALPHA[@]};do
   for seed in ${SEED[@]};do
       for beta in ${BETA[@]};do
           for mlm_w in ${MLM_W[@]};do
       log_path=/opt/data/private/noise_master/rep_ss/log/$dataset/${pre_trained_data}_noisetype_${noise_type}_noiseratio_${noise_ratio}
       rm -rf log_path
       mkdir -p $log_path
           CUDA_VISIBLE_DEVICES=$gpu_device python train_arch1.py \
               --bert_type $model --seed $seed --pre_trained_data $pre_trained_data \
               --noise_type $noise_type --noise_ratio $noise_ratio \
               --train_path $train_path --test_path $test_path  \
               --log_path $log_path/${seed}.txt \
               --batch_size 32 --learning_rate $lr  --epoch $epoch \
               --alpha $alpha --beta $beta --cl $cl --mlm_w $mlm_w --num_k $num_k --warmup_epoch $warmup_epoch \
               --show_bar
           done
       done
   done
done

noise_type=sym
noise_ratio=0.4

for alpha in ${ALPHA[@]};do
   for seed in ${SEED[@]};do
       for beta in ${BETA[@]};do
           for mlm_w in ${MLM_W[@]};do
       log_path=/opt/data/private/noise_master/rep_ss/log/$dataset/${pre_trained_data}_noisetype_${noise_type}_noiseratio_${noise_ratio}
       rm -rf log_path
       mkdir -p $log_path
           CUDA_VISIBLE_DEVICES=$gpu_device python train_arch1.py \
               --bert_type $model --seed $seed --pre_trained_data $pre_trained_data \
               --noise_type $noise_type --noise_ratio $noise_ratio \
               --train_path $train_path --test_path $test_path  \
               --log_path $log_path/${seed}.txt \
               --batch_size 32 --learning_rate $lr  --epoch $epoch \
               --alpha $alpha --beta $beta --cl $cl --mlm_w $mlm_w --num_k $num_k --warmup_epoch $warmup_epoch \
               --show_bar
           done
       done
   done
done