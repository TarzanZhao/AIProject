#export CUDA_VISIBLE_DEVICES=1,2,3,4

#python ./main.py \
#      --todo sampledata \
#      --save_folder search_10_10_5\
#      --n_log_step 1\
#      --n_save_step 10\
#      --sampleRound 1\
#      --sampleSize 20\

#python ./main.py \
#      --todo supervisedtrain \
#      --save_folder 20_search_10_10_5\
#      --n_train_data 1\
#      --load_data_folder search_10_10_5 \
#      --n_save_step 1 \
#      --n_log_step 1


#python ./main.py \
#      --todo visualize \
#      --save_folder 20_search_10_10_5\
#      --modelID 0

python ./main.py \
      --todo selfplaytrain \
      --save_folder 20_search_10_10_5\
      --overwrite 0\
      --n_save_step 3\
      --epochs 3\
      --trainround 3\
      --trainepochs 3\
      --miniTrainingEpochs 1\
      --numOfIterations 50

