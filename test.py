from nnunetv2.paths import nnUNet_results, nnUNet_raw
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch 
from batchgenerators.utilities.file_and_folder_operations import join
import argparse

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
device = torch.device('cuda')

if __name__ == "__main__":
    
    
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
        join(nnUNet_results, 'Dataset014_BrainMrs/nnUNetTrainercls__nnUNetPlans__3d_fullres'),
        use_folds=(0, ),
        checkpoint_name='checkpoint_best.pth',
    )

    predictor.predict_from_files(join(nnUNet_raw, 'Dataset014_BrainMrs/imagesTs2'),
                                    join(nnUNet_raw, 'Dataset014_BrainMrs/imagesTs_predlowres'),
                                    save_probabilities=False, overwrite=False,
                                    num_processes_preprocessing=3, num_processes_segmentation_export=3,
                                    folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
