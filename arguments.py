import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--data-path', type=str, required=True, help="path to the data files")
    parser.add_argument('--n-workers', type=int, default=8, help="number of workers in data loader")
    parser.add_argument('--mri_ratio', type=float, default=2, help="ratio for mri down-sampling")

    # diffusion model
    parser.add_argument('-t', '--timestep', type=int, default=1000, help="number of timesteps for diffusion process")
    parser.add_argument('-sche', '--beta_schedule', type=str, choices={'linear', 'sigmoid', 'cosine'}, default='cosine', help="diffusion scheduler")

    # Unet model
    parser.add_argument('-d', '--dim', type=int, default=64, help="number of feature dimensions")
    parser.add_argument('--num_heads', type=int, default=4, help="number of attention heads")
    parser.add_argument('--attn_res', type=str, default='(8, 16)', help='where the attention layers placed in the Unet.'
                                                                        'We have 4 blocks, the resolution is (1, 2, 4, 8), respectively')
    parser.add_argument('--channel_mult', type=str, default='(2, 2, 2, 2)', help='multiple ratio of channels for 4 blocks in Unet')
    parser.add_argument('--num_mod', type=int, default=4, help="number of modalities")

    # training
    parser.add_argument('-n', '--epoch', type=int, default=100, help="maximum number of epochs to train on")
    parser.add_argument('-b', '--batch-size', type=int, default=1, help="batch size fo training")
    parser.add_argument('--lr', type=float, default=0.0001, help="peak value for learning rate")
    parser.add_argument('-p', '--p_uncound', type=float, default=0.2, help="Probability of shielding modalities")
    parser.add_argument('--wd', type=float, default=0.01, help="weight decay parameter")
    parser.add_argument('--op-opt', type=str, choices={'Adam', 'AdamW'}, default='AdamW', help="optimizer option")
    parser.add_argument('--lr-opt', type=str, choices={'Off', 'StepLR', 'CosALR', 'WUStepLR', 'WUCosALR'}, default='WUCosALR', help="learning rate scheduler option")
    parser.add_argument('--lr-n', type=int, default=0, help="learning rate scheduler number")
    parser.add_argument('--wu-n', type=int, default=0, help="number of warm-up epochs")
    parser.add_argument('--accumulate-step', type=int, default=2, help="gradient accumulate steps")
    parser.add_argument('--use-amp', type=int, choices={0, 1}, default=1, help="whether to use automatic mixed precision")

    # others
    parser.add_argument('-s', '--seed', type=int, default=0, help="random seed")
    parser.add_argument('--local-rank', type=int, default=0, help='local rank for ddp settings')
    parser.add_argument('--model_path', type=str, default='./saved_models/', help="path to save or load the model")
    parser.add_argument('--image_path', type=str, default='./saved_images/', help="path to save the images")

    return parser.parse_args()
