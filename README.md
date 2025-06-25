# Clustering Analysis Using FCM and PCM on the Wine Quality Dataset

This project includes a set of scripts for performing clustering analysis on the **Wine Quality** dataset. It uses fuzzy clustering methods (FCM and PCM), Principal Component Analysis (PCA), and the Xie-Beni validity index to evaluate clustering performance.

## Project Contents

- `fcm_scr.py` – implementation of the **Fuzzy C-Means (FCM)** clustering algorithm
- `pcm_scr.py` – visualization of the **Possibilistic C-Means (PCM)** algorithm
- `pca.py` – script for performing **Principal Component Analysis (PCA)** on the input data
- `pcm.py` – implementation of the **Possibilistic C-Means (PCM)** algorithm
- `xie_beni.py` – calculation of the **Xie-Beni index**, used to evaluate the quality of clustering results

## Dataset

The analysis is based on the publicly available:  
**Wine Quality Dataset** – contains physicochemical and sensory characteristics of red wine samples.

## Requirements

- Python 3.x
- NumPy
- Pandas
- scikit-learn
- scikit-fuzzy
- sklearn
- seaborn
- Matplotlib (optional, for visualization)

## Running the Scripts

Each script can be run independently. The input data is provided in the file `winequality-red.csv`.

---

> In the future, a unified script integrating the entire analysis process is planned.

