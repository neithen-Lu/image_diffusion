# cifar 10
MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128"

torchrun --nnodes=1 --nproc_per_node=1 --master-port=29430 scripts/image_sample.py --model_path /home/qindafei/KX/image_diffusion/result/baseline_128_learnsigTrue/ema_0.9999_500000.pt --sample_path /home/qindafei/KX/image_diffusion/result/baseline_128_learnsigTrue --log_dir /home/qindafei/KX/image_diffusion/result/baseline_128_learnsigTrue $MODEL_FLAGS $DIFFUSION_FLAGS --num_samples 10000 --dependence False --gpu_no 3 --cov_type local1

# imagenet 64
# MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True"
# DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
# TRAIN_FLAGS="--lr 1e-4 --batch_size 128"

# torchrun --nnodes=1 --nproc_per_node=1 --master_port=29410 scripts/image_sample.py --model_path /home/qindafei/KX/image_diffusion/result/imagenet_base_learnsigTrue/ema_0.9999_200000.pt --sample_path /home/qindafei/KX/image_diffusion/result/imagenet_base_learnsigTrue --log_dir /home/qindafei/KX/image_diffusion/result/imagenet_base_learnsigTrue  $MODEL_FLAGS $DIFFUSION_FLAGS --num_samples 10000 --dependence False --gpu_no 1 