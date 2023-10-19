#!/bin/bash
set -e

seed=${1:-0}
vocab="data/geo/vocab.freq2.bin"
train_student_file="data/geo/train_student.bin"
train_teacher_file="data/geo/train_r2l.bin"
train_file="data/geo/train.bin"
dev_file="data/geo/test.bin"
test_file="data/geo/test.bin"
parser_student="default_student_parser"
parser="default_parser"
parser_teacher="default_teacher_parser"
dropout=0.5
hidden_size=256
embed_size=128
action_embed_size=128
field_embed_size=32
type_embed_size=32
lr_decay=0.985
lr_decay_after_epoch=20
pretrain_epoch=180
beam_size=5
lstm='lstm'
lr=0.0025
ls=0.25
max_epoch=250
patience=1000   # disable patience since we don't have dev set
src_dropout=0.5
lambda_num=0.25
model_name=model.geo.seed${seed}.dropout${dropout}.src_dropout${src_dropout}.mutual.encoder.lambda${lambda_num}.r2l

echo "**** Writing results to logs/geo/${model_name}.log ****"
mkdir -p logs/geo
echo commit hash: `git rev-parse HEAD` > logs/geo/${model_name}.log

python -u exp_distillation_share_encoder.py \
    --cuda \
    --seed ${seed} \
    --mode train \
    --batch_size 10 \
    --lambda_num ${lambda_num} \
    --asdl_file asdl/lang/lambda_dcs/lambda_asdl.txt \
    --transition_system lambda_dcs \
    --dev_file ${dev_file} \
    --train_file ${train_file} \
    --test_file ${test_file} \
    --vaildate_begin_epoch 100\
    --pretrain_epoch ${pretrain_epoch} \
    --train_student_file ${train_student_file} \
    --train_teacher_file ${train_teacher_file} \
    --parser ${parser} \
    --parser_student ${parser_student} \
    --parser_teacher ${parser_teacher} \
    --vocab ${vocab} \
    --lstm ${lstm} \
    --src_dropout ${src_dropout} \
    --primitive_token_label_smoothing ${ls} \
    --no_parent_field_type_embed \
    --no_parent_production_embed \
    --no_parent_field_embed \
    --no_parent_state \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --action_embed_size ${action_embed_size} \
    --field_embed_size ${field_embed_size} \
    --type_embed_size ${type_embed_size} \
    --dropout ${dropout} \
    --patience ${patience} \
    --max_num_trial 5 \
    --max_epoch ${max_epoch} \
    --glorot_init \
    --lr ${lr} \
    --no_copy \
    --lr_decay ${lr_decay} \
    --lr_decay_after_epoch ${lr_decay_after_epoch} \
    --decay_lr_every_epoch \
    --beam_size ${beam_size} \
    --decode_max_time_step 110 \
    --log_every 50 \
    --save_to saved_models/geo/${model_name} 2>&1 | tee -a logs/geo/${model_name}.log

. scripts/geo/test.sh saved_models/geo/${model_name}.bin 2>&1 | tee -a logs/geo/${model_name}.log