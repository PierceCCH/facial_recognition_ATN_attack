"""
Common defences against PGD attacks that we tested against our ATN model.
"""

import torch
import torch.nn.functional as F

from torchvision import transforms
from PIL import Image
from io import BytesIO

_to_pil_image = transforms.ToPILImage()
_to_tensor = transforms.ToTensor()

class Jpeg_compression(object):
    '''
    JPEG compression removes small perturbations from images by lossily compressing and decompressing them, 
    reducing the effect of adversarial noise.
    '''
    def __init__(self, device='cuda', quality=75):
        self.quality = quality
        self.device = device
    
    def __call__(self, images):
        images = self.jpegcompression(images)
        
        return images

    def jpegcompression(self, x):
        lst_img = []
        for img in x:
            img = _to_pil_image(img.detach().clone().cpu())
            virtualpath = BytesIO()
            img.save(virtualpath, 'JPEG', quality=self.quality)
            lst_img.append(_to_tensor(Image.open(virtualpath)))
            
        return torch.stack(lst_img).to(self.device)
    

class Randomization(object):
    '''
    Randomly crop, resize and pad the input images.
    '''
    def __init__(self, device='cuda', prob=0.8, crop_lst=[0.1, 0.08, 0.06, 0.04, 0.02]):
        self.prob = prob
        self.crop_lst = crop_lst
        self.device = device

    def __call__(self, images):
        images = self.input_transform(images)
        return images

    def input_transform(self, xs):
        p = torch.rand(1).item()
        if p <= self.prob:
            out = self.random_resize_pad(xs)
            return out
        else:
            return xs

    def random_resize_pad(self, xs):
        rand_cur = torch.randint(low=0, high=len(self.crop_lst), size=(1,)).item()
        crop_size = 1 - self.crop_lst[rand_cur]
        pad_left = torch.randint(low=0, high=3, size=(1,)).item() / 2
        pad_top = torch.randint(low=0, high=3, size=(1,)).item() / 2

        if len(xs.shape) == 4:
            _, _, w, h = xs.shape
        elif len(xs.shape) == 5:
            _, _, _, w, h = xs.shape
        w_, h_ = int(crop_size * w), int(crop_size * h)

        out = F.interpolate(xs, size=[w_, h_], mode='bicubic', align_corners=False)
        pad_left = int(pad_left * (w - w_))
        pad_top = int(pad_top * (h - h_))
        out = F.pad(out, [pad_left, w - pad_left - w_, pad_top, h - pad_top - h_], value=0)
        
        return out


class BitDepthReduction(object):
    '''
    Reduces the bit depth of the input images to reduce the effect of adversarial noise. 
    '''
    def __init__(self, device='cuda', compressed_bit=4):
        self.compressed_bit = compressed_bit
        self.device = device
    
    def __call__(self, images):
        images = self.bit_depth_reduction(images)
        return images

    def bit_depth_reduction(self, xs):
        bits = 2 ** self.compressed_bit
        
        xs_compress = (xs.detach() * bits).int()
        xs_255 = (xs_compress * (255 / bits))
        xs_compress = (xs_255 / 255).to(self.device)

        return xs_compress