DATASET=cnndm
EXP_NAME=window5

python inference.py \
-dataset ${DATASET} \
-exp_name transformer22/lg/${EXP_NAME} \
-cnn_ann_pkl_dir sum_dist/data/preprocess/${DATASET}-bert-ann.pkl