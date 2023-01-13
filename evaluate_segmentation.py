import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.nn.functional import interpolate
from tqdm import tqdm
import argparse
from utils import get_model

def get_voc_dataset(voc_root=None, img_size=480):
    if voc_root is None:
        voc_root = "data/voc"  # path to VOCdevkit for VOC2012
    data_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    def load_target(image):
        image = np.array(image)
        image = torch.from_numpy(image)
        return image

    target_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Lambda(load_target),
    ])

    dataset = torchvision.datasets.VOCSegmentation(root=voc_root, image_set="val", transform=data_transform,
                                                   target_transform=target_transform, download=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, drop_last=False)

    return dataset, data_loader

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
    th_attn = cum_val > (1 - args.threshold)  # Should use 0.4 to match dino
    idx2 = torch.argsort(idx)
    for head in range(nh):
        th_attn[head] = th_attn[head][idx2[head]]
    th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
    # interpolate
    th_attn = interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0]

    return th_attn


def get_per_sample_jaccard(pred, target):
    jac = 0
    object_count = 0
    for mask_idx in torch.unique(target):
        if mask_idx in [0, 255]:  # ignore index
            continue
        cur_mask = target == mask_idx
        intersection = (cur_mask * pred) * (cur_mask != 255)  # handle void labels
        intersection = torch.sum(intersection, dim=[1, 2])  # handle void labels
        union = ((cur_mask + pred) > 0) * (cur_mask != 255)
        union = torch.sum(union, dim=[1, 2])
        jac_all = intersection / union
        jac += jac_all.max().item()
        object_count += 1
    return jac / object_count


def run_eval(args, data_loader, model, device):
    model.to(device)
    model.eval()
    total_jac = 0
    image_count = 0
    for idx, (sample, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
        sample, target = sample.to(device), target.to(device)
        attention_mask = get_attention_masks(args, sample, model)
        jac_val = get_per_sample_jaccard(attention_mask, target)
        total_jac += jac_val
        image_count += 1
    return total_jac / image_count

def parse_args():
    parser = argparse.ArgumentParser(description='Transformers')

    # Loading checkpoint
    parser.add_argument("--pretraining", choices=["dino", "moco", "supervised"], help="select type of pretraining used")
    parser.add_argument("--our", action="store_true", help="set the flag to use our model, without skip connection and normalization")

    parser.add_argument("--key", choices=["student", "teacher", "base", "momentum"], default=None, help="dino/moco key to load the checkpoint")

    # data
    parser.add_argument("--voc_path", default='', type=str, required=True)
    parser.add_argument("--img_size", default=480, type=int, help="size of images")

    # segmentation evaluation arguments
    parser.add_argument('--threshold', type=float, default=0.6, help='threshold for segmentation')
    parser.add_argument('--pretrained_weights', default=None, help='pretrained weights path')
    parser.add_argument('--patch_size', type=int, default=16, help='nxn grid size of n')
    parser.add_argument('--generate_images', action='store_true', default=False, help="generate images instead of eval")

    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_dataset, test_data_loader = get_voc_dataset(voc_root=opt.voc_path, img_size=opt.img_size)


    model = get_model(opt)

    model_accuracy = run_eval(opt, test_data_loader, model, device)
    print(f"Jaccard index for {opt.pretraining}, our: {opt.our} : {model_accuracy}")