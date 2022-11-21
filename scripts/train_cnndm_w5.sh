DATASET=cnndm
EXP_NAME=window5

# stage 1
python train.py \
-dataset ${DATASET} \
-log_dir ./sum_dist/logs/transformer22/lg/${EXP_NAME} \
-load_config_dir ./sum_dist/exp_configs/0027-1.json \
-save_checkpoint_dir ./sum_dist/checkpoint/transformer22/lg/${EXP_NAME} \
-prediction_dest ./sum_dist/output/transformer22/lg/${EXP_NAME} \
-target_dest ./sum_dist/output/transformer22/lg/${EXP_NAME} \
-cnn_ann_pkl_dir ./sum_dist/data/preprocess/${DATASET}-bert-ann.pkl

# stage 2
python train.py \
-dataset ${DATASET} \
-log_dir ./sum_dist/logs/transformer22/lg/${EXP_NAME} \
-load_config_dir ./sum_dist/exp_configs/0027-2.json \
-save_checkpoint_dir ./sum_dist/checkpoint/transformer22/lg/${EXP_NAME} \
-load_checkpoint_dir ./sum_dist/checkpoint/transformer22/lg/${EXP_NAME}/checkpoint_17944.pt \
-prediction_dest ./sum_dist/output/transformer22/lg/${EXP_NAME} \
-target_dest ./sum_dist/output/transformer22/lg/${EXP_NAME} \
-cnn_ann_pkl_dir ./sum_dist/data/preprocess/${DATASET}-bert-ann.pkl

# stage 3
python train.py \
-dataset ${DATASET} \
-log_dir ./sum_dist/logs/transformer22/lg/${EXP_NAME} \
-load_config_dir ./sum_dist/exp_configs/0027-3.json \
-save_checkpoint_dir ./sum_dist/checkpoint/transformer22/lg/${EXP_NAME} \
-load_checkpoint_dir ./sum_dist/checkpoint/transformer22/lg/${EXP_NAME}/checkpoint_22430.pt \
-prediction_dest ./sum_dist/output/transformer22/lg/${EXP_NAME} \
-target_dest ./sum_dist/output/transformer22/lg/${EXP_NAME} \
-cnn_ann_pkl_dir ./sum_dist/data/preprocess/${DATASET}-bert-ann.pkl

# stage 4
python train.py \
-dataset ${DATASET} \
-log_dir ./sum_dist/logs/transformer22/lg/${EXP_NAME} \
-load_config_dir ./sum_dist/exp_configs/0027-4.json \
-save_checkpoint_dir ./sum_dist/checkpoint/transformer22/lg/${EXP_NAME} \
-load_checkpoint_dir ./sum_dist/checkpoint/transformer22/lg/${EXP_NAME}/checkpoint_26916.pt \
-prediction_dest ./sum_dist/output/transformer22/lg/${EXP_NAME} \
-target_dest ./sum_dist/output/transformer22/lg/${EXP_NAME} \
-cnn_ann_pkl_dir ./sum_dist/data/preprocess/${DATASET}-bert-ann.pkl
