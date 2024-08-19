# apptainer run --nv build/pyresidual.sif python train.py --main-seed 42 --dry-run $1 --network-conf rnn.yml --residual-conf greybox.yml

python train.py --main-seed 42 --dry-run $1 --network-conf rnn.yml --residual-conf greybox.yml --use-cuda 1