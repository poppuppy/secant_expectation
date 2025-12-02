torchrun --nnodes=1 --nproc_per_node=8 sample_ddp.py --cfg-scale 1 --embed-cfg --num-sampling-steps 1 \
--ckpt your_ckpt