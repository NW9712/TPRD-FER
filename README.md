# TPRD-FER-Official-Code-Implementations
Official Implementations of "Text Prompt Region Decomposition for Effective Facial Expression Recognition"

Our manuscript is currently under review; therefore, this project only provides inference code and weight files for testing.

The training code will be availabel later.

---
## Installation
1. Installation the package requirements
```
pip install -r requirements.txt
```

---
## Data Preparation
1. The reorganized version of [RAF-DB](http://www.whdeng.cn/RAF/model1.html) can be downloaded at [APViT](https://github.com/youqingxiaozhua/APViT):
```
data/
├─ RAF-DB/
│  ├─ basic/
│  │  ├─ EmoLabel/
│  │  │  ├─ list_patition_label.txt
│  │  │  ├─ rafdb_occlusion_list.txt
│  │  │  ├─ val_raf_db_list.txt
│  │  │  ├─ val_raf_db_list_45.txt
│  │  ├─ Image/
│  │  │  ├─ aligned_224/  # reagliend by MTCNN
```

2. The downloaded [AffectNet](http://mohammadmahoor.com/affectnet/) are organized as follow:
```
data/
├─ AffectNet/
│  ├─ Manually_Annotated_Images/
│  │  ├─ training.csv
│  │  ├─ validation.csv
│  │  ├─ 1/
│  │  │  ├─ images
│  │  │  ├─ ...
│  │  ├─ 2/
│  │  ├─ ./
```

3. The FER2013, FERPlus, and pre-processing code are available at https://github.com/microsoft/FERPlus:

4. The Occlusion- and Pose- variant lists can be downloaded at [RAN](https://github.com/kaiwang960112/Challenge-condition-FER-dataset/):

---
## Model checkpoints
- Download model checkpoints from [Google Drive](https://drive.google.com/drive/folders/17KjAjADg2mVmsbBXZlm85Eos6Iif1zMv?usp=drive_link).
- Modify the path of the downloaded weight files in the configuration files.
---
## Testing
```
python test.py --config ${CONFIG_PATH}
```


