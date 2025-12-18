
torchrun --nnodes=1 --nproc_per_node=$1 --master_port=$(shuf -i 10000-65000 -n 1) train.py --config-name=$2 ${@:3}
# accelerate launch  --num_processes $1  train_accel.py --config-name=$2 ${@:3}
