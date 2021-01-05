export CUDA_VISIBLE_DEVICES=4


python ./main.py \
      --todo experiment \
      --save_folder \
      --n_train_data 5\
      --load_data_folder stable_10_10_5\
      --trainepochs 500