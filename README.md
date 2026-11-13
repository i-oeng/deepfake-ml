# Do AI face detectors learn fakes, or shortcuts?

A course project on detecting AI-generated and manipulated faces. Getting a high score
on one dataset turned out to be easy. Keeping it on new generators and new photo
sources did not, and most of the apparent progress came from shortcuts: differences
between the real and fake training images that have nothing to do with faking a face.

This repository holds one notebook that reproduces that finding from public data.

## Run it

- **Google Colab:** open `notebooks/reproduce.ipynb`, choose a GPU runtime and run all cells.
- **Locally:** `pip install -r requirements.txt`, then open the notebook in Jupyter.

The notebook downloads about 3 GB and embeds 7,724 images. That takes about 15 minutes on
a laptop GPU (RTX 4050) and 1.5 to 2 hours on a CPU. Every download is pinned to a dataset
revision and every random choice uses one seed.

## What the notebook does

| Data | Role | Images |
|---|---|---:|
| [SFHQ-T2I](https://huggingface.co/datasets/bitmind/SyntheticFacesHighQuality-T2I) sample | AI-generated faces from FLUX.1-dev, FLUX.1-pro and SDXL | 1,724 |
| [FFHQ](https://huggingface.co/datasets/gaunernst/ffhq-1024-wds) | Real portraits, the training reals | 2,000 |
| [FairFace](https://huggingface.co/datasets/HuggingFaceM4/FairFace) | Real faces from a second dataset | 1,000 |
| [LFW](https://huggingface.co/datasets/bitmind/lfw) | Real news photos, never used in training | 1,000 |

Each image is embedded once with a frozen CLIP ViT-L/14. A logistic regression on those
embeddings is the detector. Its threshold is set on validation images so that 5% of real
faces are called fake, then left unchanged for testing.

## Results

| Question | Result |
|---|---|
| Fakes and reals of the same kind as in training | AUROC 1.000, balanced accuracy 0.983 |
| Trained on SDXL fakes only, tested on FLUX.1-dev and FLUX.1-pro | AUROC 1.000 on both |
| Real faces from datasets the detector never saw, called fake (target 5%) | FairFace 8.0%, LFW 29.7% |
| Can the embeddings tell the real datasets apart, with no fakes involved? | Yes, AUROC 1.000 for every pair |
| FairFace reals added to training | LFW faces called fake drop from 31.3% to 13.7% |

The perfect scores do not mean the detector learned what a fake looks like. All the fakes
come from one dataset and all the training reals from another, so telling the two datasets
apart is enough. A new generator from the same dataset does not expose this. A new real
dataset does: LFW news photos are called fake six times as often as the target. Training
on a second real dataset halves that, but does not fix it.

## The full project

The notebook is a small version of a larger study, which trained CLIP-based detectors over
four rounds on face-swap videos, celebrity deepfakes and photos from about 70 text-to-image
generators. It found the same pattern at scale:

- Three of the biggest apparent gains were shortcuts: the detector had learned which dataset
  a real photo came from, black borders left by face cropping, or the look of FFHQ portraits.
- Adding real photos from more sources, matching file formats and training longer helped.
- The best model still caught only 44% of OpenFake's AI faces at a 6% false-alarm rate.
  FLUX models were the hardest (AUROC 0.68 to 0.71), and the project's goal of 0.85
  balanced accuracy was not reached.

That code is not in this repository: it needs datasets that cannot be redistributed
(FaceForensics++, Celeb-DF, DF40) and days of GPU time.

## Data terms

FFHQ is CC BY-NC-SA 4.0, FairFace CC BY 4.0 and SFHQ MIT; the LFW copy used states no
license. Use the notebook's outputs for research and teaching only.
