#export CUDA_VISIBLE_DEVICES=1,2,3,4

python ./main.py \
      --todo selfplaytrain \
      --save_folder 150play\
      --overwrite 0\
      --epochs 25\
      --trainround 50\
      --trainepochs 50\
      --miniTrainingEpochs 10\
      --numOfIterations 200

