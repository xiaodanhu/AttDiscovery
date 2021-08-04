# Attribute Discovery

Attribute discovery project

## LAD Dataset
Dataset [download](https://drive.google.com/open?id=1WU2dld1rt5ajWaZqY3YLwLp-6USeQiVG), and unzip LAD_annotations and LAD_images folder under a local folder.

## CLIP Setup
Install dependencies of CLIP:

```bash
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
```

Download the CLIP model to `./checkpoints`:
```bash
$ wget "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt" -O checkpoints/model.pt
```

## Load LAD Dataset
Check `LAD_CLIP.ipynb` for how to load LAD dataset
