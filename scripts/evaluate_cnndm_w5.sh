
ROOT=sum_dist/output/inference/transformer22/lg
EXP_NAME=window5
OUTPUT_NAME=decode2

python evaluate_rouge.py \
-prediction_file ${ROOT}/${EXP_NAME}/checkpoint_31402/prediction_all-${OUTPUT_NAME}-test.txt \
-target_file ${ROOT}/${EXP_NAME}/target_all-test.txt \
-output_dir ${ROOT}/${EXP_NAME}/${OUTPUT_NAME} \
-truncate_len 50
