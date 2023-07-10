MODEL_FLAGS="--image_size 32 --num_channels 32 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 1e-4 --batch_size 32"

torchrun --nnodes=1 --nproc_per_node=4 scripts/image_train.py --data_dir /home/qindafei/KX/data/cifar $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS