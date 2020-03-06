# Action Recognition

## Setting Up

For the main code, please run the following:

```bash
conda create -n action-recognition python==3.7.6
conda activate action-recognition
pip install -r requirements.txt
```

### Downloading Video Codecs

```bash
sudo apt update
sudo apt install libdvdnav4 libdvdread4 gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly libdvd-pkg
sudo dpkg-reconfigure libdvd-pkg
sudo apt install ubuntu-restricted-extras
conda install -c conda-forge ffmpeg
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

### Feature Extraction Libraries

#### Optical Flow + Warp Optical

We use the dense flow script from https://github.com/yjxiong/dense_flow to generate the optical flow for the videos:
```bash
mkdir src/third_party && cd src/third_party
wget -O OpenCV-4.1.0.zip wget https://github.com/opencv/opencv/archive/4.1.0.zip 
unzip OpenCV-4.1.0.zip
rm OpenCV-4.1.0.zip
wget -O OpenCV_contrib-4.1.0.zip https://github.com/opencv/opencv_contrib/archive/4.1.0.zip
unzip OpenCV_contrib-4.1.0.zip
rm OpenCV_contrib-4.1.0.zip

cd opencv-4.1.0
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=ON -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.1.0/modules/ -DWITH_TBB=ON -DBUILD_opencv_cnn_3dobj=OFF -DBUILD_opencv_dnn=OFF -DBUILD_opencv_dnn_modern=OFF -DBUILD_opencv_dnns_easily_fooled=OFF ..
make -j VERBOSE=1

apt-get install libzip-dev
git clone --recursive https://github.com/yjxiong/dense_flow
cd dense_flow && mkdir build && cd build
OpenCV_DIR=../../opencv-4.1.0/build/  cmake ..
make -j
```
