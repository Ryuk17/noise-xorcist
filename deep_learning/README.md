# A code template for training DNN-based speech enhancement models.
A training code template is highly valuable for deep learning engineers as it can significantly enhance their work efficiency. Despite different programmers have varying coding styles, some are excellent while others may not be as good. My philosophy is to prioritize simplicity. In this context, I am sharing a practical organizational structure for training code files in speech enhancement (SE). The primary focus is on keeping it concise and intuitive rather than aiming for comprehensiveness.

## 🔥 News
- [**2025-3-31**] Added a new branch named `plus` for better implementation. Please use this one directly.
- [**2024-5-28**] Added a new branch named `pro` for better implementation.

## File Specification
* `configs`: Configuration files for training and infernce.
* `DNSMOS`: Pre-trained DNSMOS checkpoints from Microsoft.
* `evaluation`: Metric calculation scripts adapted from URGENT 2024 official scripts.
* `models`: Model definitions.
* `prepare_datasets`: Scripts for generating DNS3 training data.
* `dataloader.py`: Dataset class for the dataloader.
* `distributed_utils.py`: Distributed Data Parallel (DDP) training utils.
* `evaluate.py`: Evaluation script based on scp files obtained by inference.
* `infer.py`: Inference script.
* `loss_factory.py`: Various useful loss functions in SE.
* `scheduler.py`: Warmup scheduler definition.
* `train.py`: Training script, surpporting both multiple-GPU and single-GPU conditions.

## Usage
When starting a new SE project, you should follow these steps:
1. Modify `dataloader.py` to match your dataset;
2. Define your own model in `models`;
3. Modify the `configs/cfg_train` to match your training setup;
4. Select a loss function in `loss_factory.py`, or create a new one if needed;
5. Run `train.py`:
   ```
   python train.py
   python train.py -D 1
   python train.py -C configs/cfg_train.yaml -D 1
   python train.py -C configs/cfg_train.yaml -D 0,1,2,3
   ```
6. After training finished, specify your checkpoint and paths in `configs/cfg_infer.yaml`;
7. Run `evaluate.py`.

## Note
1. The code is originally intended for Linux systems, and if you attempt to adapt it to the Windows platform, you may encounter certain issues:
* Incompatibility of paths: The file paths used in Linux systems may not be compatible with the file paths in Windows.
* Challenges in installing the pesq package: The process of installing the pesq package on Windows may not be straightforward and may require additional steps or configurations.

2. Thanks for starring if you find this repo useful.

## Acknowledgement
This code template heavily borrows from the excellent [Sheffield_Clarity_CEC1_Entry](https://github.com/TuZehai/Sheffield_Clarity_CEC1_Entry) repository in many aspects.
