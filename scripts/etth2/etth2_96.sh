
model_name=FreqMixAttNet
run_date='test'
root_path='./data'


pred_len=96
learning_rate=0.2
dropout=0.1

sr_ratio= 4
n_heads=8
d_model=16
freq_weight=8
weight_att=0.03
seq_len=384

aug_weight=0.05
batch_size=128
alpha=0.7 

e_layers=2
l1l2_alpha=0.00
d_ff=32
train_epochs=6
patience=6
down_sampling_layers=2
down_sampling_window=2
mix_rate=0.1
jitter_ratio=0.3
devices='0'
decomp_method='wavelet'

python -u run_model.py \
--gpu 0 \
--task_name long_term_forecast \
--is_training 1 \
--devices $devices \
--data_path ETTh1.csv \
--root_path $root_path \
--model_id $run_date'_ETTh1' \
--model $model_name \
--data ETTh1 \
--features M \
--seq_len $seq_len \
--label_len 0 \
--pred_len $pred_len \
--e_layers $e_layers \
--decomp_method wavelet \
--enc_in 7 \
--c_out 7 \
--des 'Exp' \
--itr 1 \
--patch_len 16 \
--d_model $d_model \
--n_heads $n_heads \
--weight_att $weight_att \
--sr_ratio $sr_ratio \
--d_ff $d_ff \
--down_sampling_layers $down_sampling_layers \
--down_sampling_window $down_sampling_window \
--levels 3 \
--freq_weight $freq_weight \
--alpha $alpha \
--l1l2_alpha $l1l2_alpha \
--learning_rate $learning_rate \
--dropout $dropout \
--train_epochs $train_epochs \
--patience $patience \
--batch_size $batch_size \
--aug_weight $aug_weight \
--mix_rate $mix_rate \
--jitter_ratio $jitter_ratio 
