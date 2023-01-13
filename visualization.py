import colorsys
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from skimage.measure import find_contours
from torch.nn.functional import interpolate
from tqdm import tqdm
import argparse
import cv2
from utils import get_model

def unnoramlize(img, scale=False):
    if len(img.shape) == 4:
        img = img[0]
    if torch.is_tensor(img):
        img = img.detach().permute(1,2,0).numpy()
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    out = np.empty_like(img)
    for c in range(3):
        out[:, :, c] = img[:, :, c] * std[c] + mean[c]

    if scale:
        out = (out * 255).astype('uint8')
    return out

def get_attention_masks(args, image, model):
    # make the image divisible by the patch size
    w, h = image.shape[2] - image.shape[2] % args.patch_size, image.shape[3] - image.shape[3] % args.patch_size
    image = image[:, :w, :h]
    w_featmap = image.shape[-2] // args.patch_size
    h_featmap = image.shape[-1] // args.patch_size

    attentions = model.get_last_selfattention(image.cuda())['attn']
    nh = attentions.shape[1]

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    # we keep only a certain percentage of the mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cum_val = torch.cumsum(val, dim=1)
    th_attn = cum_val > (1 - args.threshold)
    idx2 = torch.argsort(idx)
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head]]
    th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
    # interpolate
    th_attn = interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0]

    return th_attn

def apply_mask_last(image, mask, color=(0.0, 0.0, 1.0), alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image

def display_instances(image, mask, path, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):

    image = image.permute(1, 2, 0).cpu().numpy()
    mask = mask.cpu().numpy()

    plt.ioff()
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]

    # Generate random colors
    def random_colors(N, bright=True):
        """
        Generate random colors.
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        return colors

    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = (image * 255).astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            pass
            # _mask = cv2.blur(_mask, (10, 10))
        # Mask
        masked_image = apply_mask_last(masked_image, _mask, color, alpha)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    plt.close(fig)
    masked_image = masked_image.astype(np.uint8)
    plt.imsave(fname=os.path.join(path, fname), arr=masked_image)

def display_save_blobs(ccl, fname) :

    background = (0, 0, 0)  # [235, 235, 235]
    # background = (235, 235 , 235 )
    h, w = ccl.shape
    colors = [background,
              (254, 102, 13), (8, 96, 168),
              (128, 166, 206), (166, 218, 149),
              (245,255,250), (238,130,238),
              (220,20,60), (240,255,255)]

    # colors = random_colors(n_blob, bright=False)
    x = ccl.cpu().numpy()
    labels = np.unique(x)
    out = np.empty((h, w, 3))
    for lb, color in zip(labels, colors) :
        if lb == 0 :
            color = np.array(background)  # np.ones(3)
        mask = x == lb
        out[mask] = np.array(color)
    out = out.astype('uint8')
    plt.imsave(fname, out)

def generate_images_per_model(args, model, device):

    model.to(device)
    model.eval()
    data_transform_model = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    data_transform_clean = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()])

    samples = []
    images = []
    names = []
    for im_name in tqdm(os.listdir(args.test_dir)):
        im_path = f"{args.test_dir}/{im_name}"
        img = Image.open(f"{im_path}").convert('RGB')

        img_th = data_transform_model(img)
        samples.append(img_th)

        img_clean = data_transform_clean(img)
        images.append(img_clean)

        names.append(im_name.split('.')[0])

    samples = torch.stack(samples, 0).to(device)
    images = torch.stack(images, 0).to(device)

    attention_masks = []
    for sample in samples:
        attention_masks.append(get_attention_masks(args, sample.unsqueeze(0), model))

    save_path = os.path.join(args.save_path, f'{args.model_name}_{args.threshold}_{args.img_size}')
    os.makedirs(save_path, exist_ok=True)

    for idx, (image, mask, name) in enumerate(zip(images, attention_masks, names)):
        plt.imsave(os.path.join(save_path, f"{name}.png"), (image.permute(1,2,0).cpu().numpy() * 255).astype('uint8'))
        for head_idx, mask_h in enumerate(mask):
            f_name = f"{name}_{head_idx}.png"
            display_instances(image, mask_h, path=save_path, fname=f_name, alpha=args.alpha)


def parse_args():
    parser = argparse.ArgumentParser(description='Transformers')

    # Loading checkpoint
    parser.add_argument("--pretraining", choices=["dino", "moco", "supervised"], help="select type of pretraining used")
    parser.add_argument("--our", action="store_true", help="set the flag to use our model, without skip connection and normalization")

    parser.add_argument("--dino_checkpoint_key", choices=["student", "teacher"], default=None, help="dino key to load the checkpoint")
    parser.add_argument("--moco_checkpoint_key", choices=["base", "momentum"], default="momentum",
                        help="moco key to load the checkpoint")

    # Images
    parser.add_argument("--test_dir", type=str, default='')
    parser.add_argument("--save_path", type=str, default='')
    parser.add_argument("--img_size", type=int, default=480)

    # segmentation evaluation arguments
    parser.add_argument('--threshold', type=float, default=0.6, help='threshold used for segmentation, it is 1 - val')
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--pretrained_weights', default=None, help='pretrained weights path')
    parser.add_argument('--patch_size', type=int, default=16, help='nxn grid size of n')

    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_args()
    opt.model_name = opt.pretraining + '_' + str(opt.our)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = get_model(opt)

    generate_images_per_model(opt, model, device)
