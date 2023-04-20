import sys
sys.path.append(".")
import numpy as np
import torch
from torch.autograd import Function
import torchvision
from PIL import Image
import argparse
import os



def iou(outputs: np.array, labels: np.array):
    
    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)


    return iou.mean()

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).to(device = input.device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


def eval_seg(pred,true_mask_p,threshold = (0.1, 0.3, 0.5, 0.7, 0.9)):
    '''
    threshold: a int or a tuple of int
    masks: [b,1,h,w]
    pred: [b,1,h,w]
    '''
    b, c, h, w = pred.size()

    eiou, edice = 0,0

    for th in threshold:

        gt_vmask_p = (true_mask_p > th).float()
        vpred = (pred > th).float()
        vpred_cpu = vpred.cpu()
        disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')

        disc_mask = gt_vmask_p[:,0,:,:].squeeze(1).cpu().numpy().astype('int32')
        
        '''iou for numpy'''
        eiou += iou(disc_pred,disc_mask)

        '''dice for torch'''
        edice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()
        
    return eiou / len(threshold), edice / len(threshold)

def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--inp_pth")
    argParser.add_argument("--out_pth")
    args = argParser.parse_args()
    mix_res = (0,0)
    num = 0
    pred_path = args.inp_pth
    gt_path = args.out_pth
    for root, dirs, files in os.walk(pred_path, topdown=False):
        for name in files:
            if 'ens' in name:
                num += 1
                strr = name.split('.jpg')[0]
                print(strr)
                pred = Image.open(os.path.join(root, name)).convert('L')
                # tumor_seg_training_data_data_segmentation_v2_test_groundTruth_2_2_00377_sub0_0_region13_4_Pos
                gt_name = strr+'.png'
                gt = Image.open(os.path.join(gt_path, gt_name)).convert('L')
                pred = torchvision.transforms.PILToTensor()(pred)
                pred = torch.unsqueeze(pred,0).float() 
                pred = pred / pred.max()
               
                gt = torchvision.transforms.PILToTensor()(gt)
                gt = torchvision.transforms.Resize((256,256))(gt)
                gt = torch.unsqueeze(gt,0).float() / 255.0
              
                temp = eval_seg(pred, gt)
                iou, dice = temp[0],temp[1]
                print('individual iou is',iou)
                print('individual dice is', dice)
                mix_res = tuple([sum(a) for a in zip(mix_res, temp)])
                print(mix_res)
    iou, dice = tuple([a/num for a in mix_res])
    print('combined iou is',iou)
    print('combined dice is', dice)

if __name__ == "__main__":
    main()
