# **Cellcounter**

**Live demo:** https://sallamsaka-cellcounter.hf.space  

Cellcounter predicts a cell density map from microscopy images using a U-Net model and a total cell count from the sum of the density map.

---

## **Web App Summary**
- **Input:** an image (preferably an actual microscopy image)
- **Output:**
  - Predicted density map
  - Predicted cell count

**How to use:** click a gallery image or upload your own image and inference runs immediately.

---

## **Brief Methodology**
- **Model:** U-Net style CNN trained to output a density map using PyTorch.
- **Training data:** 2018 Data Science Bowl nuclei dataset (Kaggle). 80% of the stage1 training data was used for training and 20% for validation.
- **Label construction:** for each cell mask, I computed a centroid and placed a normalized Gaussian around each centroid so that each cell mask image array sums to one and the sum of all cell masks is the total number of cells.
- **Data Augmentation:** Doubled the training set with contrast-inverted images. Training was performed on cropped out 256x256 patches of images to cover each image.
- **Loss incorporated the following factors:**
  - Pixel-level value deviation from target density
  - Shape of predicted densities at different scales
  - Absolute error of total counts
- **Inference:** sliding-window prediction on 256×256 patches, then stitch patches into a full-image density map.

---

## **Results**
I also compared a count-only training loss vs a multi-term (pixel + shape + count) loss.

**Count-only loss**
- Patch MAE: 2.615
- Full-image MAE: 6.366

**Multi-term (pixel/shape/count) loss**
- Patch MAE: 3.021
- Full-image MAE: 5.209

**Takeaway:** count-only training did slightly better on patches, but the multi-term loss produced better full-image behavior after stitching probably due to the spatial information provided by the predicted density.

---

## **Examples**
- **Example 1:**
<p align="center">
  <img src="screenshots/Screenshot1.png" width="600" />
</p>

- **Example 2:**
<p align="center">
  <img src="screenshots/Screenshot2.png" width="600" />
</p>

---

## **Limitations**
The model performs best on:
- cells that contrast well with the background
- uniform cell sizes/appearance

The model seems to struggle with:
- overlapping / crowded cells
- large variation in cell size
- low contrast images
- variation in transparency of cells
- colorful / complex looking cells

The Kaggle dataset was not designed specifically for density prediction, so, some inconsistencies in the data such as fractions of cells being considered a whole cell leads to inconsistencies in training.

---

## **Repository structure**
```text
.
├── app.py
├── best.pt
├── demo_images/
├── screenshots/
├── requirements.txt
├── cell count.ipynb
├── count_loss_trial.ipynb
└── README.md
```

---

## **Citations / credits**

- Allen Goodman, Anne Carpenter, Elizabeth Park, jlefman-nvidia, Josette_BoozAllen, Kyle, Maggie, Nilofer, Peter Sedivec, and Will Cukierski. 2018 Data Science Bowl . https://kaggle.com/competitions/data-science-bowl-2018, 2018. Kaggle.
- ChatGPT (learning aid, guidance, troubleshooting, etc)
