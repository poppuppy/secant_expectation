# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from accelerate import Accelerator

from models import DiT_models
from models_original import DiT_models as Teacher
# from diffusion import create_diffusion
from transport import create_transport, Sampler
from train_utils import parse_transport_args


#################################################################################
#                             Training Helper Functions                         #
#################################################################################


class CustomDataset(Dataset):
    def __init__(self, features_dir):
        # features_dir, _features/_labels
        L = os.listdir(features_dir)
        print(f'---> Folders in {features_dir}: {L}')
        for name in L:
            if name.endswith('_features'):
                self.features_dir = os.path.join(features_dir, name)
            elif name.endswith('_labels'):
                self.labels_dir = os.path.join(features_dir, name)


        self.features_files = sorted(os.listdir(self.features_dir), key=lambda x:int(x.split('_')[0])*8+int(x[-5]))[:-1]
        self.labels_files = sorted(os.listdir(self.labels_dir), key=lambda x:int(x.split('_')[0])*8+int(x[-5]))[:-1]
        print(len(self.features_files), len(self.features_files))
        # assert len(self.features_files) == len(self.features_files) == 1281167 # ImageNet

    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]

        features = np.load(os.path.join(self.features_dir, feature_file))
        labels = np.load(os.path.join(self.labels_dir, label_file))
        return torch.from_numpy(features).squeeze(0), torch.from_numpy(labels).squeeze(0)


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


# class CustomDataset(Dataset):
#     def __init__(self, features_dir, labels_dir):
#         self.features_dir = features_dir
#         self.labels_dir = labels_dir

#         self.features_files = sorted(os.listdir(features_dir))
#         self.labels_files = sorted(os.listdir(labels_dir))

#     def __len__(self):
#         assert len(self.features_files) == len(self.labels_files), \
#             "Number of feature files and label files should be same"
#         return len(self.features_files)

#     def __getitem__(self, idx):
#         feature_file = self.features_files[idx]
#         label_file = self.labels_files[idx]

#         features = np.load(os.path.join(self.features_dir, feature_file))
#         labels = np.load(os.path.join(self.labels_dir, label_file))
#         return torch.from_numpy(features), torch.from_numpy(labels)


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup accelerator:
    accelerator = Accelerator()
    device = accelerator.device

    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        # experiment_index = len(glob(f"{args.results_dir}/*"))
        # model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        # experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        experiment_dir = f"{args.results_dir}/{args.exp_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        online_cfg=args.online_cfg,
    )
    # Note that parameter initialization is done within the DiT constructor
    model = model.to(device)
    
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    teacher = Teacher[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    ).to(device)

    state_dict = torch.load(args.teacher_ckpt, map_location=lambda storage, loc: storage)
    teacher.load_state_dict(state_dict)
    del state_dict
    requires_grad(teacher, False)
    teacher.eval()

    if args.auto_resume and args.ckpt is None:
        try:
            existing_checkpoints = os.listdir(checkpoint_dir)
            if len(existing_checkpoints) > 0:
                existing_checkpoints.sort()
                args.ckpt = os.path.join(checkpoint_dir,
                                         existing_checkpoints[-1])
        except Exception:
            pass
        if args.ckpt is not None:
            logger.info(f"Auto resuming from: {args.ckpt}")
    
    if args.ckpt is not None:
        ckpt_path = args.ckpt
        state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        # state_dict = find_model(ckpt_path)
        model.load_state_dict(state_dict["model"])
        ema.load_state_dict(state_dict["ema"])
        # del state_dict
        # opt.load_state_dict(state_dict["opt"])
        # args = state_dict["args"]
    elif args.init_from is not None:
        state_dict = torch.load(args.init_from, map_location=lambda storage, loc: storage)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        del state_dict
        logger.info(f"Model initialized. Missing keys: {missing_keys}, unexpected keys: {unexpected_keys}")
        ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    requires_grad(ema, False)
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )  # default: velocity; 
    # diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    if accelerator.is_main_process:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(args.lr)
    if args.ckpt is not None:
        opt.load_state_dict(state_dict["opt"])

    # Setup data:
    # features_dir = f"{args.feature_path}/imagenet256_features"
    # labels_dir = f"{args.feature_path}/imagenet256_labels"
    # dataset = CustomDataset(features_dir, labels_dir)
    dataset = CustomDataset(args.feature_path)
    loader = DataLoader(
        dataset,
        # batch_size=int(args.global_batch_size // accelerator.num_processes),
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} images ({args.feature_path})")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    # model, teacher, opt, loader = accelerator.prepare(model, teacher, opt, loader)
    model, teacher, opt, loader = accelerator.prepare(model, teacher, opt, loader)

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_epoch = 0
    if args.ckpt is not None:
        train_steps = int(existing_checkpoints[-1].split('.')[0])
        start_epoch = int(train_steps / int(1281167 / args.global_batch_size))
    
    start_time = time()
    
    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device).squeeze(0)
            y = y.to(device).squeeze(0)
            # x = x.squeeze(dim=1)
            # y = y.squeeze(dim=1)
            model_kwargs = dict(y=y)
            loss_dict = transport.training_losses(model, teacher, x, args.online_cfg, args.num_steps, args.loss_type, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            update_ema(ema, model)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = accelerator.reduce(avg_loss, reduction='sum')
                avg_loss = avg_loss.item() / accelerator.num_processes
                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    
    if accelerator.is_main_process:
        logger.info("Done!")


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", type=str, default="features")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--exp-name", type=str, default="exp")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--init-from", type=str, default=None)
    parser.add_argument("--loss-type", type=str, choices=["sdei", "stei", "sdee", "stee"], default="sdei")
    parser.add_argument("--online-cfg", type=str, default=None)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--teacher-ckpt", type=str, default="", help="Teacher DiT checkpoint")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a DiT checkpoint")
    parser.add_argument("--no-auto-resume", action="store_false", dest="auto_resume")

    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)
