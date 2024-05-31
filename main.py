import os
import time
import random

import numpy as np
import torch
import torchio as tio
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.distributed as dist
import matplotlib.pyplot as plt

from arguments import parse_arguments
from data.load import load_data, preprocess, load_subjs_batch
from model.unet3d import UnetModel
from model.diffusion_model import GaussianDiffusion

if __name__ == "__main__":
    args = parse_arguments()
    args.device_id = args.gpus = args.local_rank
    args.attn_res = ast.literal_eval(args.attn_res)
    args.channel_mult = ast.literal_eval(args.channel_mult)
    args.device = torch.device('cuda:' + str(args.device_id))
    args.use_amp = args.use_amp == 1

    # seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # path
    if not os.path.exists(args.image_path):
        os.mkdir(args.image_path)
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    # ddp
    dist.init_process_group(backend='nccl', rank=args.device_id)

    # dataset
    transform = [tio.ToCanonical(), tio.CropOrPad(target_shape=(192, 192, 144))]
    transform += [tio.Resample(target=(args.mri_ratio, args.mri_ratio, args.mri_ratio))]
    transform += [tio.ZNormalization()]
    transform = tio.Compose(transform)

    # preload
    x, y = load_data(data_path=args.data_path)
    dataset = tio.SubjectsDataset(list(x), transform=transform)
    data_loader = DataLoader(dataset, num_workers=8)
    x = preprocess(data_loader)
    del dataset, data_loader
    dataset = tio.SubjectsDataset(list(x), transform=None)
    sampler_train = torch.utils.data.DistributedSampler(dataset)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.n_workers,
                              sampler=sampler_train, pin_memory=True, drop_last=False)

    # model
    model = UnetModel(
        in_channels=1,
        model_channels=args.dim,
        out_channels=1,
        channel_mult=args.channel_mult,
        attention_resolutions=args.attn_res,
        num_mod=args.num_mod
    )
    if args.model_path and os.path.exists(args.model_path) and args.model_path.endswith('.h5'):
        model = torch.load(args.model_path)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(args.device_id)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.device_id])
    gaussian_diffusion = GaussianDiffusion(timesteps=args.timestep, beta_schedule=args.beta_schedule)

    # op
    if args.op_opt == 'Adam':
        optimizer = optim.Adam([{'params': model.module.parameters(), 'lr': args.lr, 'weight_decay': args.wd}])
    elif args.op_opt == 'AdamW':
        optimizer = optim.AdamW([{'params': model.module.parameters(), 'lr': args.lr, 'weight_decay': args.wd}])

    # scheduler
    if 'WU' in args.lr_opt and args.wu_n <= 0:
        args.wu_n = args.epoch // 5
    if 'StepLR' in args.lr_opt:
        if args.lr_n <= 0:
            args.lr_n = args.epoch // 10
        scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_n)
    elif 'CosALR' in args.lr_opt:
        if args.lr_n <= 0:
            args.lr_n = args.epoch - args.wu_n if 'WU' in args.lr_opt else args.epoch
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.lr_n)
    if 'WU' in args.lr_opt:
        scheduler0 = optim.lr_scheduler.LambdaLR(optimizer, lambda e: (e + 1) / args.wu_n)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, [scheduler0, scheduler], [args.wu_n])

    # scaler
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    # train
    len_data = len(data_loader)
    for epoch in range(args.epoch):
        model.train()
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
            time_start = time.time()
            for step, subjs_batch in enumerate(data_loader):
                images, _, _ = load_subjs_batch(subjs_batch)
                B, M, H, W, D = images.shape
                assert M == args.num_mod, f"Except number of modalities args.num_mod = {args.num_mod}, but get {M}"
                images = images.view(B * M, 1, H, W, D)
                labels = torch.tensor([0, 1, 2, 3] * B).long()
                batch_size = images.shape[0]
                images = images.to(args.device, non_blocking=True)
                labels = labels.to(args.device, non_blocking=True)
                batch_mask = (torch.rand(batch_size) > args.p_uncound).int().to(args.device)  # random mask for modality labels
                t = torch.randint(0, args.timestep, (batch_size,), device=args.device).long()  # sample t uniformally
                # forward
                loss = gaussian_diffusion.train_losses(model, images, t, labels, batch_mask)
                if (step + 1) % max(len_data // 10, 1) == 0 or (step + 1) == len_data:
                    time_end = time.time()
                    print("Epoch{}/{}\t  Step{}/{}\t Loss {:.4f}\t Time: {:.2f}".format(epoch + 1, args.epoch, step + 1, len_data, loss.item(), time_end - time_start))
                    time_start = time_end
                scaler.scale(loss).backward()
                if (step + 1) % args.accumulate_step == 0 or step == len_data - 1:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
        if (epoch + 1) % 10 == 0:
            model.eval()
            # generate
            n_sample = 4
            ddim_step = 50
            ddim_all = []
            for i in range(n_sample):
                ddims = gaussian_diffusion.ddim_sample(model, image_size=(1, H, W, D), batch_size=args.num_mod, ddim_timesteps=ddim_step,
                                                       n_class=args.num_mod, w=2.0, mode='all', ddim_discr_method='quad', ddim_eta=0.0,
                                                       clip_denoised=False)
                ddim_all.append(np.concatenate([np.expand_dims(d, axis=0) for d in ddims], axis=0))  # (ddim_step + 1, num_mod, 1, H, W, D)
            ddim_all = np.concatenate([np.expand_dims(d, axis=0) for d in ddim_all], axis=0)  # (n_sample, ddim_step + 1, num_mod, 1, H, W, D)
            ddim_all = np.swapaxes(ddim_all, 0, 1)  # (ddim_step + 1, n_sample, num_mod, 1, H, W, D)
            medium_axis = D // 2
            imgs = ddim_all[-1].reshape(n_sample, args.num_mod, H, W, D)[..., medium_axis] # (ddim_step + 1, n_sample, num_mod, 1, H, W)
            # plot
            fig = plt.figure(figsize=(12, 5), constrained_layout=True)
            gs = fig.add_gridspec(n_sample, args.num_mod)
            for n_row in range(n_sample):
                for n_col in range(args.num_mod):
                    f_ax = fig.add_subplot(gs[n_row, n_col])
                    f_ax.imshow((imgs[n_row, n_col] + 1.0) * 255 / 2, cmap="gray")
                    f_ax.axis("off")
            plt.savefig(os.path.join(args.image_path, f'DDIM_w={2.0}_all.png'))
            torch.save(model, args.model_path + f'mri_dm_epoch={epoch}.h5')
        
        if (epoch + 1) == args.epoch:
            torch.save(model, args.model_path + 'mri_dm_last.h5')
