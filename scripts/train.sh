TRAIN_POKEMON(){
model_type='resnet18'
lr=1e-4
devices=0
datapath='/data1/su/pokemon/pokemon_dataset/'
datathread=4
trainlist="/data1/su/transfer_learning/filenames/pokemon_train.txt"
vallist="/data1/su/transfer_learning/filenames/pokemon_val.txt"
train_batch_size=32
val_batch_size=1
outf="saved_models"
startEpoch=0
total_epochs=100
logFile="logs/pokemon_res18"
scale_size="[224,224]"

cd ..
CUDA_VISIBLE_DEVICES=0  python train.py \
                        --model_type $model_type \
                        --lr $lr \
                        --devices $devices \
                        --datathread $datathread \
                        --datapath $datapath \
                        --trainlist $trainlist \
                        --vallist $vallist \
                        --train_batch_size $train_batch_size \
                        --val_batch_size $val_batch_size \
                        --outf $outf \
                        --startEpoch $startEpoch \
                        --total_epochs $total_epochs \
                        --logFile $logFile \
                        --scale_size $scale_size



}

TRAIN_POKEMON