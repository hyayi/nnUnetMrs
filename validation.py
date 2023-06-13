import os 
import numpy as np
import nibabel as nib
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero,create_nonzero_mask
from acvl_utils.cropping_and_padding.bounding_boxes import get_bbox_from_mask, crop_to_bbox, bounding_box_to_slice

from nnunetv2.run.run_training import get_trainer_from_args
import torch
import argparse

def validate(args):
    model_args = {
    "dataset_name_or_id" : args.dataset_name_or_id,
    "configuration" : '3d_fullres',
    "fold" : args.fold,
    "trainer_name" : 'nnUNetTrainercls',
    "plans_identifier" : 'nnUNetPlans',
    "use_compressed" : False,
    "device" : torch.device('cuda'),
    }
    trainer = get_trainer_from_args(**model_args)
    trainer.initialize()
    trainer.load_checkpoint(args.checkpoint_path)
    trainer.perform_actual_validation(True)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_path", type=str, default="/study/nnUnet_an/nnUNet_results/Dataset015_BrainMrs/nnUNetTrainercls__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--dataset_name_or_id", type=str, default="15")
    args = parser.parse_args()
    
    validate(args)