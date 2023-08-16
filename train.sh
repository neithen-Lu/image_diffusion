# cifar-10
MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128"

torchrun --nnodes=1 --nproc_per_node=1 --master_port=29430 scripts/image_train.py --data_dir /home/qindafei/KX/data/cifar $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS --dependence False --log_dir result/baseline_128_learnsigTrue --cov_type local2 --gpu_no 3

# imagenet 64
# MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True"
# DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
# TRAIN_FLAGS="--lr 1e-4 --batch_size 128"

# torchrun --nnodes=1 --nproc_per_node=1 --master_port=29420 scripts/image_train.py --data_dir /home/qindafei/KX/data/imagenet_train_64x64 $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS --dependence False --log_dir result/imagenet_base_learnsigTrue --cov_type local2 --gpu_no 2