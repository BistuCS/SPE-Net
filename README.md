# SPE-Net: Structural Perception Enhancement for Cross-View Geo-Localization

This repository provides the official (or reference) implementation for **Structural Perception Enhancement (SPE)** for **Cross-View Geo-Localization (CVGL)**, focusing on matching **UAV (drone-view)** images with **satellite-view** images under drastic viewpoint and layout variations.

## Abstract
Cross-view geo-localization (CVGL) presents significant challenges due to the drastic variations in perspective and scene layout between unmanned aerial vehicle (UAV) and satellite images. Existing methods primarily emphasize global semantic feature extraction, but often overlook fine-grained local regions and struggle to align cross-view features, limiting their ability to capture discriminative target information. To address this issue, we propose a **Structural Perception Enhancement (SPE)** network for CVGL. Built upon the **DINOv2** backbone, the network integrates a **Local Region Mining Module (LRMM)** for extracting discriminative regional features and enabling accurate cross-view feature alignment. Furthermore, we introduce a **Sample Rebalancing Strategy (SRS)** to address training instability caused by satellite image scarcity and sample imbalance. Extensive experiments on the **University-1652** and **SUES-200** datasets show that our method surpasses existing state-of-the-art approaches, with average improvement of **0.44% in R@1** and **0.96% in mAP/AP**, validating its effectiveness and superiority.

**Keywords:** Cross-view geo-localization · Image retrieval · UAVs · Remote sensing

---

## Method Overview

SPE-Net consists of three major components:

- **DINOv2 Backbone (shared weights):** Extracts global token features and patch-level local tokens for both drone and satellite images.
- **Local Region Mining Module (LRMM):**
  - **RDM (Region Disentanglement Module):** Separates foreground/background representations via patch-level foreground probability estimation.
  - **FAM (Feature Alignment Module):** Projects global/foreground/background features into a shared embedding space to mitigate cross-view misalignment.
- **Sample Rebalancing Strategy (SRS):** Adds an auxiliary rebalancing constraint to alleviate training instability caused by satellite-sample scarcity and view imbalance.

> The overall objective combines **InfoNCE loss** for cross-view alignment and an auxiliary **sample rebalancing loss** for stability:
> \[
> \mathcal{L} = \mathcal{L}_{InfoNCE} + 0.1 \mathcal{L}_{SR}
> \]

---

## Results

SPE-Net achieves strong performance on public CVGL benchmarks:

- **University-1652:** improvements over prior SOTA methods with stronger cross-view alignment.
- **SUES-200:** consistent gains across multiple flight altitudes, showing robustness under low-altitude UAV imaging.

(You can add tables/figures from your paper here if you want.)

---

## Requirements

We provide a `requirements.txt` to ensure compatibility.

```bash
pip install -r requirements.txt
```

---

## Data Preparation

This repo does **not** include datasets. Please download datasets and organize them in your local directory.

### Supported Datasets
- **University-1652**
- **SUES-200**

> Note: Dataset paths and structure may be configured in the dataset loaders under `sample4geo/dataset/`.

### (Optional) Pretrained Weights
If you use pretrained weights (e.g., DINOv2 / trained checkpoints), place them under a dedicated folder such as `pretrained/` (or follow the paths used in your scripts/configs).

---

## Training

Typical training scripts in this repo include (examples):
- `train_university.py`
- `train_sues.py`

Example:
```bash
python train_university.py
```

```bash
python train_sues.py
```

---

## Evaluation

Evaluation scripts include (examples):
- `eval_university.py`
- `eval_sues.py`

Example:
```bash
python eval_university.py
```

```bash
python eval_sues.py
```

---

## Project Structure (Typical)

```
SPE-Net/
├─ pretrained/                 # (optional) pretrained weights / checkpoints
├─ data/                       # datasets (ignored by git)
├─ sample4geo/                 # core training / model code (backbone, lrmm, loss, trainer, datasets)
├─ train_university.py
├─ train_sues.py
├─ eval_university.py
├─ eval_sues.py
├─ requirements.txt
└─ README.md
```

---

## Citation

If you find SPE-Net useful, please cite our paper:

```bibtex
@inproceedings{SPE-Net,
  title     = {Structural Perception Enhancement for Cross-View Geo-Localization},
  author    = {<Your Name> and <Coauthors>},
  booktitle = {<Conference/Journal>},
  year      = {20XX}
}
```

---

## Acknowledgements

This work builds upon prior research in cross-view geo-localization and uses **DINOv2** as the backbone. We thank the authors of related open-source projects and benchmark datasets.

---

## License

Add your license information here (if applicable).