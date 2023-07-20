MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma False --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128"

torchrun --nnodes=1 --nproc_per_node=1 --master-port=29450 scripts/image_train.py --data_dir /home/qindafei/KX/data/cifar $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS --dependence True --log_dir result/decay0.1_128 