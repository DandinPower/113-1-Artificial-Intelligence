SAVE_FOLDER=cnn
BATCH_SIZE=8
TEST_BATCH_SIZE=1024
TRAIN_VALID_SPLIT=0.8
EPOCHS=60
LR=0.0005
GAMMA=0.85
SEED=42

python main.py --save_folder $SAVE_FOLDER --batch_size $BATCH_SIZE --test_batch_size $TEST_BATCH_SIZE --train_valid_split $TRAIN_VALID_SPLIT \
    --epochs $EPOCHS --lr $LR --gamma $GAMMA --seed $SEED