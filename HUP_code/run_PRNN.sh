#!/usr/bin/env bash

model_tag="PRNN"

device="cuda1"

#rnn_type="GRU"
#rnn_type="LSTM"
#rnn_type="TimeLSTM"
rnn_type="BLSTM"
RNN_norm="LSTM"

bottom_emb_item_len=5


folder_type="Applicances"
#folder_type="Computers"

folder_base='/export/sdb/home/guyulong/program/HUP/HUP_Data'
folder_data_input=""
folder_data_model='Model/Model_'
folder_data_log="Log/Log_"

exp_note='TimeLSTM'
exp_note='LSTM'
exp_note='BLSTM'



if [ ${folder_type} = "Computers" ]; then
    seq_len=29
    train_len=577507
    train_len_div=3700157
    test_len=246967
    file_flag_train_test="session.SBCGD.id.len30.SBCGD.id"
    file_flag="session.SBCGD.id.len30.SBCGD.id"
    file_mapping="session.SBCGD.id.len30.SBCGD.mapping"
    init_emb_wgt_path='sku.reidx,bh.reidx,cid3.reidx,gap.reidx,dwell.reidx'
fi


if [ ${folder_type} = "Applicances" ]; then
    seq_len=39
    train_len=583282
    train_len_div=11178791
    test_len=250776
    file_flag_train_test="session.SBCGD.id.len40.SBCGD.id"
    file_flag="session.SBCGD.id.len40.SBCGD.id"
    file_mapping="session.SBCGD.id.len40.SBCGD.mapping"
    init_emb_wgt_path='sku.reidx,bh.reidx,cid3.reidx,gap.reidx,dwell.reidx'
fi


#input
folder_data_input_type="${folder_data_input}${folder_type}"

#output
#folder_data_output_type="${folder_data_output}${folder_type}"

#Model/Model
folder_data_model_type="${folder_data_model}${model_tag}"

#Log/Log_PRNN
folder_data_log_type="${folder_data_log}${model_tag}"

folder_input="${folder_base}/${folder_data_input_type}"


folder_model="${folder_data_model_type}/${folder_type}"


folder_log="${folder_base}/${folder_data_log_type}/${folder_type}"

#output data
folder_output="${folder_base}/${folder_model}"

flag_train_model=1

flag_train_use_div=0

flag_test_candidate_all=1

batch_size=128
train_epoch=2
flag_embedding_trainable=1

loss_weights_micro_sku_cid3="0,0.5,0.5"
loss_weights_sku_cid3=${loss_weights_micro_sku_cid3}
drop_out_r=0.0
mode_attention=2
#mode_attention=0
att_layer_cnt=2
bhDwellAtt=1
layer_num=3
#set micro emb: which type to connect

rnn_state_size="100,100,100"

if [ ${flag_train_use_div} = "1" ]; then
    train_len=${train_len_div}
else
    train_len=${train_len}
fi


file_train="${file_flag_train_test}.train"
if [ ${flag_train_use_div} = "1" ]; then
    file_train="${file_flag_train_test}.train.div"
fi
file_test="${file_flag_train_test}.test"


flag_file_train_test_len="Traindiv${flag_train_use_div}_TrainTestLen${train_len}_${test_len}"

model_flag="Layer${layer_num}_State${rnn_state_size}_Bot${bottom_emb_item_len}_EmbT${flag_embedding_trainable}_LosW${loss_weights_sku_cid3}_Att${mode_attention}_AttLay${att_layer_cnt}_bdatt${bhDwellAtt}_Dp${drop_out_r}_Epo${train_epoch}_Bat${batch_size}_${flag_file_train_test_len}"

model_flag="${model_tag}_${model_flag}_${rnn_type}_${RNN_norm}"


file_model="Model_${file_flag}_${model_flag}.model"

exp_sig="${file_flag}_train_${model_flag}_CandiAll${flag_test_candidate_all}"

file_log_base="${folder_log}/log_${exp_sig}"
file_log_train="${file_log_base}_train"
file_log_test="${file_log_base}_test"


file_log=${file_log_train}
if [ ${flag_train_model} = "0" ]; then
    file_log=${file_log_test}
fi


if [ -n "${exp_note}" ]; then
    file_model="${file_model}_${exp_note}"
    file_log="${file_log}_${exp_note}"
fi



cmd="THEANO_FLAGS='device=${device}' nohup python -u PRNNRec.py ${model_tag} ${layer_num} ${rnn_state_size} ${bottom_emb_item_len} ${flag_embedding_trainable} ${folder_input} ${folder_output} ${exp_sig} ${init_emb_wgt_path} ${file_train} ${file_test} ${train_len} ${test_len} ${seq_len} ${batch_size} ${file_model} ${file_mapping} top1000sku sku.mapping ${flag_train_model} ${train_epoch} ${flag_test_candidate_all} ${mode_attention} ${drop_out_r} ${loss_weights_sku_cid3} ${att_layer_cnt} ${bhDwellAtt} ${rnn_type} ${RNN_norm} > ${file_log} &"
echo ${cmd}

eval ${cmd}
