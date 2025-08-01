
This is the anonymous repository corresponding to the AAAI AISI 2026 submission: "LEMON: A Large Endoscopic MONocular Dataset and Foundation Model for Perception in Surgical Settings". **This repository is intended for AAAI AISI 2026 reviewers only.**

 
## 🔥🔥🔥 LEMON Dataset
To facilitate the download, sharing, and review of the dataset, we have significantly reduced its size to a more manageable level. Specifically, we have downsampled videos to 1 frame per second (fps), downsized their resolution to below 640x360 pixels, and stored them in an LMDB format (approximately 400GB of storage space). You can download LEMON dataset and LemonFM foundation model by [this link](https://mega.nz/folder/GVkgDQKZ). 
**The password to decrypt it is included in the first section of the supplementary material submitted for review.**


The video annotation file can be downloaded here: [labels.json](https://github.com/anonymous-8989/sub-156/blob/main/labels.json)
 
### 🎥 Demo


<div align="center">
  <img src="https://github.com/user-attachments/assets/79f25335-6d8c-4b9b-afe1-59cfd5c84a39" width="70%" > </img>
</div>



### 🛠️ Dependencies and Installation
Install the following dependencies in your local setup:

   ```bash
   $ git clone git@github.com:anonymous-8989/sub-156.git
   $ cd sub-156 && pip install -r requirements.txt
   ```

### 🔗 Load Dataset 

Using the following code to load the LEMON dataset LMDB and its annotation file:
```python
import lmdb
import numpy as np
import cv2
import json

# Load annotation
with open('labels.json', 'r') as file:
   annotation = json.load(file)
for video in annotation:
   video_id = video['youtubeId']
   procedure_type = video['procedureName']
   surgery_type = 'robotic' if video['robotic'] else 'non-robotic'


# Load LEMON dataset
# Open the LMDB environment in read-only mode
env = lmdb.open('lmdb_path', readonly=True, lock=False, readahead=False)
with env.begin() as txn:
   cursor = txn.cursor()
   for key, value in tqdm(cursor, total = txn.stat()['entries']):
        # Decode the image name from bytes to string
        img_name = key.decode('utf-8')
        img_array = np.frombuffer(value, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

```

## 🎉🎉🎉 LemonFM Foundation Model 

You can download the LemonFM checkpoint within the provided link with the password.

### 🛠️ LemonFM Training

Follow the provided scripts to launch your own LemonFM training.

```bash
$ python3 -m torch.distributed.run --nproc_per_node=8 --nnodes=1 LemonFM/LemonFM.py --arch convnext_large --data_path 'LEMON dataset lmdb path' --output_dir 'your path to store the trained foundation model' --batch_size_per_gpu 24 --num_workers 10
```

### 🚀 LemonFM inference
The code below shows how to run our Surg-FM foundation model to get a feature vector for any given surgical video frame:

   ```python
   import torch
   from PIL import Image
   from src.model_loader import build_LemonFM

   # Load the pre-trained LemonFM model
   LemonFM = build_LemonFM(pretrained_weights = 'your path to the LemonFM')
   LemonFM.eval()

   # Load the image and convert it to a PyTorch tensor
   img_path = 'path/to/your/image.jpg'
   img = Image.open(img_path)
   img = img.resize((224, 224))
   img_tensor = torch.tensor(np.array(img)).unsqueeze(0).to('cuda')

   # Extract features from the image using the ResNet50 model
   outputs = LemonFM(img_tensor)
   ```


### 🚀 Example of fine-tuning LemonFM for surgical phase recognition

```bash
$ python3 downstream/train_phase_recognition_autolaparo.py --lr 1e-3 --opt adamW --nepochs 100 --bs 512 --cpdir 'path/to/store/checkpoint' --logdir 'path/to/store/log' --lmdb 'path/to/downstream_task/lmdb' --labels 'path/to/downstream_task/annotation' --seed 30 --pretrained-weights 'path/to/our/LemonFM.pth'
```

```bash
$ python3 downstream/test_phase_recognition_autolaparo.py --lmdb 'path/to/downstream_task/lmdb' --models 'path/to/your/cpdir' --labels 'path/to/downstream_task/annotation'
```


## 🏃‍♂️🏃‍♂️🏃‍♂️ Recreate LEMON dataset From Scratch

Alternatively, you can recreate our LEMON dataset from scratch to acquire the complete LEMON videos.


### 🧱 Data Curation Pipeline

You can use our code of the data curation pipeline and provided annotation file (["*labels.json*"](https://github.com/anonymous-8989/sub-156/blob/main/labels.json)) to recreate the whole LEMON dataset.

1. Get your Youtube cookie:

   You need to provide a "cookies.txt" file if you want to download videos that require Youtube login. 

   Use the [cookies](https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp) extension to export your Youtube cookies as "cookies.txt".


2. Download the annotation file (["*labels.json*"](https://github.com/anonymous-8989/sub-156/blob/main/labels.json)) and use the video downloader to download the original selected Youtube videos.

   ```bash
   $ python3 src/video_downloader.py --video-path '../labels.json' --output 'your path to store the downloaded videos' --cookies 'your YouTube cookie file'
   ```

3. Curate the downloaded original videos as LEMON video dataset. In detail, use the video_processor to classify each frame as either 'surgical' or 'non-surgical', then remove the beginning and end segments of non-surgical content from the videos, and mask the non-surgical regions in 'surgical' frames and the entire 'non-surgical' frames.

   ```bash
   $ python3 src/video_processor.py --input 'your original downloaded video storage path' --input-json '../labels.json' --output 'your path to store the curated videos and their corresponding frame annotation files' --classify-models 'frame classification model' --segment-models 'non-surgical object detection models'
   ```

4. Process the LEMON video dataset as LEMON image dataset (For foundation model pre-training).

   ```bash
   $ python3 src/create_lmdb.py --video-folder 'your directory containing the curated videos and their corresponding frame annotation files' --output-json 'your path for the json file to verify the videos and labels alignment' --lmdb-path 'your lmdb storage path'
   ```
