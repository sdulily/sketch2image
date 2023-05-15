# End-to-End Freehand Sketch-to-Image Generation 

## Preparation

### 1. Create virtual environment (optional)
All code was developed and tested on Windows 10  with Python 3.8.0 (Anaconda) and PyTorch 1.8.1.

```bash
$ conda create -n sketch2image python=3.8.0
$ conda activate sketch2image
```

### 2. Clone the repository
```bash
$ git clone git@github.com:sdulily/sketch2image.git
$ cd End-to-End-Freehand-Sketch-to-Image-Generation
```
complete model and dataset downlown at here:
https://download.csdn.net/download/artistkeepmonkey/87784790
### 3. Install dependencies
```bash
$ pip install -r requirements.txt
```

### 4. Download datasets

The datasets are located in `datasets/`
### 5. Download pretrained models
The pretrained model is located in `/checkpoints/pretrained/`
### Test models

Test on the shoes dataset:
```bash
$ python test.py --dataroot ./datasets/shoes --pretrained
```

## Contact

If you encounter any problems, please contact us.
## Reference
Our project borrows some source files from CycleGAN(https://junyanz.github.io/CycleGAN.git). 
