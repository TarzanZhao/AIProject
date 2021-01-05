#export CUDA_VISIBLE_DEVICES=1,2,3,4

python ./main.py \
      --todo sampledata \
      --save_folder notstable_10_10_5\
      --n_log_step 1\
      --n_save_step 3\
      --sampleRound 1\
      --sampleSize 10