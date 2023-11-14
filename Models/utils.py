import sys
import math
import torch
from Models.model import CNN
from Utils.console import console

def lip_clip(model:torch.nn.Module, clip:float):

    with torch.no_grad():
        i = 1
        for n, p in model.named_parameters():
            if ('weight' in n) & ('last_lay' not in n):
                if 'cnn_layers' in n:
                    conv_filter = p.data.detach().clone()
                    out_ch, in_ch, h, w = conv_filter.shape
                    
                    transpose1 = torch.transpose(conv_filter, 1, 2)
                    matrix1 = transpose1.reshape(out_ch*h, in_ch*w)
                    
                    transpose2 = torch.transpose(conv_filter, 1, 3)
                    matrix2 = transpose2.reshape(out_ch*w, in_ch*h)

                    matrix3 = conv_filter.view(out_ch, in_ch*h*w)

                    transpose4 = torch.transpose(conv_filter, 0, 1)
                    matrix4 = transpose4.reshape(in_ch, out_ch*h*w)

                    norm_1 = torch.linalg.matrix_norm(matrix1, ord=2).item()
                    norm_2 = torch.linalg.matrix_norm(matrix2, ord=2).item()
                    norm_3 = torch.linalg.matrix_norm(matrix3, ord=2).item()
                    norm_4 = torch.linalg.matrix_norm(matrix4, ord=2).item()

                    norm = h * min([norm_1, norm_2, norm_3, norm_4])
                else:
                    norm = torch.linalg.matrix_norm(p.data, ord=2).item()
                p.data = p.data / (norm + 1e-12)

    return model

def clip_weight(model:torch.nn.Module, clip:float):
    norm = torch.linalg.matrix_norm(model.last_lay.weight.data, ord=2).item()
    if norm > clip:
        model.last_lay.weight.data = model.last_lay.weight.data / (norm + 1e-12)
    return model

def check_clipped(model:torch.nn.Module, clip:float):

    res = True
    with torch.no_grad():
        i = 1
        for n, p in model.named_parameters():
            if ('weight' in n):
                if 'cnn_layers' in n:
                    conv_filter = p.data.detach().clone()
                    out_ch, in_ch, h, w = conv_filter.shape
                    
                    transpose1 = torch.transpose(conv_filter, 1, 2)
                    matrix1 = transpose1.reshape(out_ch*h, in_ch*w)
                    
                    transpose2 = torch.transpose(conv_filter, 1, 3)
                    matrix2 = transpose2.reshape(out_ch*w, in_ch*h)

                    matrix3 = conv_filter.view(out_ch, in_ch*h*w)

                    transpose4 = torch.transpose(conv_filter, 0, 1)
                    matrix4 = transpose4.reshape(in_ch, out_ch*h*w)

                    norm_1 = torch.linalg.matrix_norm(matrix1, ord=2).item()
                    norm_2 = torch.linalg.matrix_norm(matrix2, ord=2).item()
                    norm_3 = torch.linalg.matrix_norm(matrix3, ord=2).item()
                    norm_4 = torch.linalg.matrix_norm(matrix4, ord=2).item()

                    norm = math.sqrt(h*w) * min([norm_1, norm_2, norm_3, norm_4])
                else:
                    norm = torch.linalg.matrix_norm(p.data, ord=2).item()
                cond = (norm - clip) > 1e-5
                if cond:
                    console.log(f'[bold][red] Failed initial clip check :x:: clip is {clip} and norm is {p.norm(p=2).item()}, at layer: {n}')
                    sys.exit()
    if res:
        console.log('[bold][green] Pass initial clip check: :white_check_mark:')
    return res

def init_model(args):
    model = CNN(channel=[32, 64], hid_dim=[256, 64], img_size=args.img_sz, channel_in=args.channel_in, out_dim=args.num_class, kernal_size=3, debug=args.debug)
    return model
