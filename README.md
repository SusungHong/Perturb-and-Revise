# [CVPR 2025] Perturb-and-Revise: Flexible 3D Editing with Generative Trajectories

Susung Hong, Johanna Karras, Ricardo Martin-Brualla, and Ira Kemelmacher-Shlizerman

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://susunghong.github.io/Perturb-and-Revise/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2412.05279)

![Demonstration](https://susunghong.github.io/Perturb-and-Revise/assets/split_gifs/comp_part18.gif)

## Installation

- Follow the installation instructions for [MVDream-threestudio](https://github.com/bytedance/MVDream-threestudio)
- We have tested with:
  - PyTorch 2.0.1+cu[version]
  - torchvision 0.15.2+cu[version]
  - Python 3.9

## Usage

### Quick Start

Run the test script:
```bash
bash test_pnr.sh
```

### Generate a Synthetic Object
```bash
python launch.py --config configs/mvdream-sd21-shading-schedule.yaml --train --gpu 0 \
    system.prompt_processor.prompt="$ORIG_PROMPT" \
    system.background.eval_color=[1.0,1.0,1.0] \
    tag="$ORIG_PROMPT" use_timestamp=false name="synthetic"
```

### Edit an Existing Object
Choose the appropriate config:
- `mvdream-pnr.yaml`: For general objects
- `mvdream-pnr-synthetic.yaml`: For synthetic objects (includes a resolution milestone)

```bash
python launch.py --config configs/mvdream-pnr.yaml --train --gpu 0 \
    system.prompt_processor.prompt="$EDIT_PROMPT" \
    system.background.eval_color=[1.0,1.0,1.0] \
    system.weights="outputs/synthetic/$ORIG_PROMPT/ckpts/last.ckpt" \ # Replace this with the path to your weights
    tag="$EDIT_PROMPT" use_timestamp=false name="pnr"
```

### IPG Steps
```bash
python launch.py --config configs/mvdream-pnr-ipg.yaml --train --gpu 0 \
    system.prompt_processor.prompt="$EDIT_PROMPT" \
    system.background.eval_color=[1.0,1.0,1.0] \
    system.weights="outputs/synthetic/$ORIG_PROMPT/ckpts/last.ckpt" \ # Replace this with the path to your weights
    system.edit_weights="outputs/pnr/$EDIT_PROMPT/ckpts/last.ckpt" \
    tag="$EDIT_PROMPT" use_timestamp=false name="pnr_ipg"
```

### More Parameters to Consider

- **`trainer.max_steps` and other step-related parameters** 
  - For first-stage editing without IPG, add 50 steps to your desired count
  - This accounts for the 50 steps used to determine the perturbation value
  - Also applies to parameters like `resolution_milestones`, `min_step_percent`, and `max_step_percent`

- **`system.auto_eta_upper_bound`**
  - Controls the upper limit for perturbation value (Default: 0.6)
  - For less perturbation, decrease this value

## Citation

```bibtex
@article{hong2024perturb,
  title={Perturb-and-Revise: Flexible 3D Editing with Generative Trajectories},
  author={Hong, Susung and Karras, Johanna and Martin-Brualla, Ricardo and Kemelmacher-Shlizerman, Ira},
  journal={arXiv preprint arXiv:2412.05279},
  year={2024}
}
```

## Acknowledgements

This code builds upon the following repositories:
- [threestudio](https://github.com/threestudio-project/threestudio)
- [MVDream-threestudio](https://github.com/bytedance/MVDream-threestudio)

We thank the authors for their open-source contributions that made this work possible.
