import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel
import torchvision.utils as vutils
from torchvision import transforms
from PIL import Image
class OurDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform, test_flag=False):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.transform = transform

        self.test_flag=test_flag
        
        
        x_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            transforms.Resize(256)
        ])
        y_transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5]),
            transforms.Resize(256)
        ])
        
        self.norm_x_transform = x_transforms
        self.norm_y_transform = y_transforms

        
        sample_dirs = ['/home/jupyter/MedSegDiff/gcs/tumor_seg_training_data/data/segmentation_v2/train/img/1',
                       '/home/jupyter/MedSegDiff/gcs/tumor_seg_training_data/data/segmentation_v2/train/img/2',
                       '/home/jupyter/MedSegDiff/gcs/tumor_seg_training_data/data/segmentation_v2/train/img/3']
        
        paths = []
        for sample_dir in sample_dirs:
            base_path = os.listdir(sample_dir)
            for file_name in base_path:
                if file_name.endswith(".png"):
                    paths.append(os.path.join(sample_dir, file_name))
                
        
        gt_dirs = ['/home/jupyter/MedSegDiff/gcs/tumor_seg_training_data/data/segmentation_v2/train/groundTruth/1',
                       '/home/jupyter/MedSegDiff/gcs/tumor_seg_training_data/data/segmentation_v2/train/groundTruth/2',
                       '/home/jupyter/MedSegDiff/gcs/tumor_seg_training_data/data/segmentation_v2/train/groundTruth/3']
        
        gt_paths = []
        for gt_dir in gt_dirs:
            base_path = os.listdir(gt_dir)
            for file_name in base_path:
                if file_name.endswith(".png"):
                    gt_paths.append(os.path.join(gt_dir, file_name))
        
        # self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.sample_list = sorted(paths)
        self.gt_list=sorted(gt_paths)
        
        
#         if test_flag:
#             self.seqtypes = ['t1', 't1ce', 't2', 'flair']
#         else:
#             self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']

#         self.seqtypes_set = set(self.seqtypes)
#         self.database = []
#         for root, dirs, files in os.walk(self.directory):
#             # if there are no subdirs, we have data
#             if not dirs:
#                 files.sort()
#                 datapoint = dict()
#                 # extract all files as channels
#                 for f in files:
#                     seqtype = f.split('_')[3]
#                     datapoint[seqtype] = os.path.join(root, f)
#                 assert set(datapoint.keys()) == self.seqtypes_set, \
#                     f'datapoint is incomplete, keys are {datapoint.keys()}'
#                 self.database.append(datapoint)

    def __getitem__(self, x):
        if self.test_flag:
            path = self.sample_list[x]
            gt_path=self.gt_list[x]

            img = Image.open(path)
            gt = Image.open(gt_path)

            img = self.norm_x_transform(img)
            gt = self.norm_y_transform(gt)
            # mask = Image.open(gt).convert('L')
            if self.transform:
                image = self.transform(img)
                
           
            return (image, image, path)
        
        else:
            path = self.sample_list[x]
            gt_path=self.gt_list[x]

            img = Image.open(path)
            gt = Image.open(gt_path)

            img = self.norm_x_transform(img)
            gt = self.norm_y_transform(gt)
            # mask = Image.open(gt).convert('L')
            if self.transform:
                image = self.transform(img)
            
            return (image, gt, path)
            

        
#         out = []
#         filedict = self.database[x]
#         for seqtype in self.seqtypes:
#             nib_img = nibabel.load(filedict[seqtype])
#             path=filedict[seqtype]
#             out.append(torch.tensor(nib_img.get_fdata()))
#         out = torch.stack(out)
#         if self.test_flag:
#             image=out
#             image = image[..., 8:-8, 8:-8]     #crop to a size of (224, 224)
#             if self.transform:
#                 image = self.transform(image)
#             return (image, image, path)
#         else:

#             image = out[:-1, ...]
#             label = out[-1, ...][None, ...]
#             image = image[..., 8:-8, 8:-8]      #crop to a size of (224, 224)
#             label = label[..., 8:-8, 8:-8]
#             label=torch.where(label > 0, 1, 0).float()  #merge all tumor classes into one
#             if self.transform:
#                 state = torch.get_rng_state()
#                 image = self.transform(image)
#                 torch.set_rng_state(state)
#                 label = self.transform(label)
#             return (image, label, path)

    def __len__(self):
        return 5495


