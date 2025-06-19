# UWT-Net
## 📰News
**[2025.6.18]** UWT-Net is accepted by MICCAI-25. 

## Datasets
Link to original dataset:
BUSI:[data](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
GLAS:[data](https://websignon.warwick.ac.uk/origin/slogin?shire=https%3A%2F%2Fwarwick.ac.uk%2Fsitebuilder2%2Fshire-read&providerId=urn%3Awarwick.ac.uk%3Asitebuilder2%3Aread%3Aservice&target=https%3A%2F%2Fwarwick.ac.uk%2Ffac%2Fcross_fac%2Ftia%2Fdata%2Fglascontest&status=notloggedin)
CVC-ClinicDB:[data](https://www.kaggle.com/datasets/balraj98/cvcclinicdb)

Also, we provide all the [pre-processed dataset](https://1drv.ms/f/c/bb0380f07d1dc124/EqWdcferptlNsQGxftXujWABA4u8_J6RfEId96QMmUrqZA?e=FgwjIt) without requiring any further data processing. You can directly download and put them into the data dir.
The resulted file structure is as follows.
```
UWT-Net
├── inputs
│   ├── BUSI
│     ├── images
│           ├── malignant (1).png
|           ├── ...
|     ├── masks
│        ├── 0
│           ├── malignant (1)_mask.png
|           ├── ...
│   ├── GLAS
│     ├── images
│           ├── 0.png
|           ├── ...
|     ├── masks
│        ├── 0
│           ├── 0.png
|           ├── ...
│   ├── CVC-ClinicDB
│     ├── images
│           ├── 0.png
|           ├── ...
|     ├── masks
│        ├── 0
│           ├── 0.png
|           ├── ...
```
## How To Use
```javascript
conda create -n uwtnet python=3.10 anaconda   
```
```javascript
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116  
```
```javascript
pip install -r requirements.txt  
```
## Evaluating UWT-Net
We provide pre-trained models [checkpoint](https://1drv.ms/f/c/bb0380f07d1dc124/ErAyQwjA-nNEtoZxaIjcr7kBtmYMAfKVA_mXDbyUosCwtg?e=VdZ9Df) for all datasets that are our best results in the experiments. Note that this is different from the average results of the paper.
|Dataset|IoU|F1|Checkpoint|
|-|-|-|-|
|BUSI|67.14|80.17|[link](https://1drv.ms/f/c/bb0380f07d1dc124/ElhMrEBvq0RKvmbzG2jSawMBUvJQx35gtvf-UfGkpv8gJw?e=iaHS2g)|
|GLAS|88.54|93.92|[link](https://1drv.ms/f/c/bb0380f07d1dc124/EgtJ1DcJZCFOsz07JmgOF-cBo2hpRUqIRD3PEC46_A73Aw?e=H7atfK)|
|CVC-ClinicDB|87.63|93.38|[link](https://1drv.ms/f/c/bb0380f07d1dc124/EgtJ1DcJZCFOsz07JmgOF-cBo2hpRUqIRD3PEC46_A73Aw?e=qxggdE)|

 Run the following scripts to evaluate
 ```javascript
python val.py --name ${dataset}_UWTNet --output_dir [YOUR_OUTPUT_DIR]
```
For example, evaluate UWT-Net with the resolution of 256x256 with a single GPU on the BUSI dataset.
```javascript
python val.py  --name busi_UWTNet --output_dir ./VAL
```
## Train UWT-Net
You can simply train UWT-Net on a single GPU by specifing the dataset name and input size.
```javascript
python train.py  --dataset {dataset} --input_w {input_size} --input_h {input_size} --name {dataset}_UWTNet  --data_dir [YOUR_DATA_DIR]
```
For example, train UWT-Net with the resolution of 256x256 with a single GPU on the BUSI dataset.
```javascript
python train.py  --dataset busi --input_w 256 --input_h 256 --name busi_UWTNet  --data_dir ./inputs
```
Note that the resolution of glas is 512x512, differing with other datasets (256x256).
## References
[WTConv](https://github.com/BGU-CS-VIL/WTConv)

[UKAN](https://github.com/CUHK-AIM-Group/U-KAN)

