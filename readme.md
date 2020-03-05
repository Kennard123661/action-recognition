# Action Recognition

## Setting Up

For the main code, please run the following:

```bash
conda create -n action-recognition python==3.7.6
conda activate action-recognition
pip install -r requirements.txt
```

### Datasets

#### Kinetics
To download kinetics dataset, create `{DATA_DIR}/kinetics/splits` and place the downloaded `.json` and `.csv` 
downloaded from [Kinetics-400](https://deepmind.com/research/open-source/kinetics) into the folder.

```bash
# {split} is one of ['train', 'test', 'val']
python src/extras/download_kinetics400.py --split {split} 
```

#### ActivityNet

To download activity-net 1.3, download `youtube-dl` first:
```bash
sudo apt-get install youtube-dl
```
Then, run the following:
```bash
python src/extras/create_anet_download_script.py
cd {DATA_DIR}/activitynet/
bash ./download-anet13.sh
```
