<div align="center">
<h1>GMambaPolyp</h1>
<h3>GMambaPolyp:  A State-Space Model Network with Global Context Awareness and Noise-Robust Feature Fusion for Polyp Segmentation</h3>

## 🛠Setup

#### 1. Mamba preparation

Down the mamba code from [Google Driver](https://drive.google.com/drive/folders/1BYnSyR3Ck1qJt0xZv02UaPnQiBOh_mLL?usp=drive_link) and move it into `./lib/`.

```html
GMambaPolyp
├── lib
├── ├── vmamba
├── ├── ├── kernels
├── ├── ├── mamba2
├── ├── ├── vmunet.py
├── ├── ├── ...
├── ├── model.py
├── utils
```

#### 2. Environment

```python
conda create -n gmamba-polyp python=3.10
conda activate gmamba-polyp
cd GMambaPolyp
pip install -r requirements.txt
```

## 📚Data Preparation

Downloading training and testing datasets and move them into `./data/`.

**TrainDatasets**: The dataset can be found [here](https://drive.google.com/drive/folders/1NVEDXDeIvKHw55dOnL6CbbbsiWrg41FH?usp=drive_link).

**TestDatasets**: The dataset can be found [here](https://drive.google.com/drive/folders/12i58jDzDGE8MiQ-QxPxiltbX8GkzwaG4?usp=drive_link).

```html
GMambaPolyp
├── data
├── ├── TrainDataset
├── ├── ├── images
├── ├── ├── masks
├── ├── ├── edges
├── ├── TestDataset
├── ├── ├── Kvasir
├── ├── ├── CVC-ClinicDB
├── ├── ├── CVC-300
├── ├── ├── CVC-ColonDB
├── ├── ├── ETIS-LaribPolypDB
```

## ⏳Training

```python
python train.py
```

## 🔖Testing

```python
python test.py
```





