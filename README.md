# Attribute Discovery

Attribute discovery project
```
├── checkpoints             # Saved models and processed data
  └── model.pt              # Downloaded CLIP model (see instruction below)
├── data                    # Datasets
  ├── wiki                  # Wiki Data
  └── LAD                   # LAD dataset
    ├── LAD_annotations     
    └── LAD_images          
└── AttDiscovery            # Project files
  ├── clip                  # Scripts from CLIP
  ├── LAD_CLIP.ipynb        # Use CLIP to extract features for classification on LAD dataset
  └── README.md
```

## LAD Dataset
Dataset [Homepage](https://github.com/PatrickZH/A-Large-scale-Attribute-Dataset-for-Zero-shot-Learning)

Dataset [Download](https://drive.google.com/open?id=1WU2dld1rt5ajWaZqY3YLwLp-6USeQiVG), and unzip `LAD_annotations` and `LAD_images` folder under a local folder `LAD`.

## CLIP Setup
Install dependencies of CLIP:

```bash
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
```

Download the CLIP model to `./checkpoints`:
```bash
$ wget "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt" -O ../checkpoints/model.pt
```

## Load LAD Dataset
Check `LAD_CLIP.ipynb` for how to load LAD dataset
