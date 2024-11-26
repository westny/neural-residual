# Description: Run the training and testing of the model

#### Train ####
python train.py --main-seed 42 --dry-run $1 --network-conf rnn.yml --residual-conf greybox.yml --use-cuda 1  # --balance-data 1

## Run with Apptainer:
# apptainer run --nv build/pyresidual.sif python train.py --main-seed 42 --dry-run $1 --network-conf rnn.yml --residual-conf greybox.yml

#### Test ####
# python test.py --main-seed 42 --dry-run $1 --network-conf rnn.yml --residual-conf greybox.yml --use-cuda 1