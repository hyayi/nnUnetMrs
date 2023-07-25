import os 
import warnings 
warnings.filterwarnings("ignore")
from nnunetv2.paths import nnUNet_results, nnUNet_raw
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch 
from batchgenerators.utilities.file_and_folder_operations import join
import argparse
from time import time

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
device = torch.device('cuda')

def test(args):
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True,
        device=torch.device('cuda'),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
        )


    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, args.model_path),
        use_folds=args.folds,
        checkpoint_name=args.checkpoint_name,
    )
    os.makedirs(join(nnUNet_raw, args.save_path), exist_ok=True)
    predictor.predict_from_files(join(nnUNet_raw, args.input_path),
                                    join(nnUNet_raw,args.save_path),
                                    save_probabilities=False, overwrite=True,
                                    num_processes_preprocessing=1, num_processes_segmentation_export=1,
                                    folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, default="/study/nnUnet_an/nnUNet_raw/Dataset015_BrainMrs/imagesTs")
    parser.add_argument("--save_path", type=str, default="Dataset015_BrainMrs/imagesTs_predlowres")
    parser.add_argument("--checkpoint_name", type=str, default="checkpoint_final.pth")
    parser.add_argument("--model_path", type=str, default="Dataset015_BrainMrs/nnUNetTrainercls__nnUNetPlans__3d_fullres")
    parser.add_argument("--folds", nargs='+',default=(0,1,3,4))  
    args = parser.parse_args()
    
    test(args)
