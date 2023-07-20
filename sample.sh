MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 1e-4 --batch_size 32"

torchrun --nnodes=1 --nproc_per_node=1 --master-port=29400 scripts/image_sample.py --model_path /home/qindafei/KX/image_diffusion/checkpoint/cifar10_uncond_50M_500K.pt --sample_path /home/qindafei/KX/image_diffusion/checkpoint $MODEL_FLAGS $DIFFUSION_FLAGS --num_samples 10000 --dependence False --gpu_no 2