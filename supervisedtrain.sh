#export CUDA_VISIBLE_DEVICES=1,2,3,4

python ./main.py \
      --todo sampledata \
      --save_folder stable_10_10_5\
      --n_log_step 1\
      --n_save_step 3\
      --sampleRound 1\
      --sampleSize 50

#      --probDepth0 (0.0,0.0,0.5,0.5)
#      --probDepth1 (0.1,0.4,0.2,0.3)

#      --epsilon0 (0.02,0.07) \
#      --epsilon1 (0.1,0.2) \


#python ./main.py \
#      --todo supervisedtrain \
#      --save_folder 20_search_10_10_5\
#      --n_train_data 1\
#      --load_data_folder search_10_10_5 \
#      --n_save_step 1 \
#      --n_log_step 1


#python ./main.py \
#      --todo selfplaytrain \
#      --save_folder 20_search_10_10_5\
#      --overwrite 0\
#      --n_save_step 3\
#      --epochs 3\
#      --trainround 3\
#      --trainepochs 3\
#      --miniTrainingEpochs 1\
#      --numOfIterations 50

