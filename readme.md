# Action Recognition

## Setting Up
The directories for our codebase is setup as follows:
```bash
[CS5242-PROJECT-DIR]
    - [CS5242-PROJECT-CODE] # this repository
    - datasets
        - breakfast
            - mstcn  # download from mstcn repository
                - groundTruth
                - splits
                - features
            - i3d  # provided i3d features
            - i3d-2048  # i3d features from breakfast dataset 
            - labels  # labels from breakfast dataset
            - videos  # videos from breakfast dataset
            - segment.txt  # download from kaggle
            - provided-gt  # download from kaggle database
            - splits # download from kaggle dataset
        - activitynet
        - ...
    - submissions       
```
Then, edit the root directory inside of `[CS5242-PROJECT-CODE]/src/config.py`, to link to `[CS5242-PROJECT-DIR]`. You 
may place `[CS5242-PROJECT-CODE]` in a separate directory.

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

#### Installing nvidia optical flow directory

First, we install the dependencies for each of the following modules for compiling opencv from source.

```bash
sudo apt-get install build-essential 
sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev

# for images
sudo apt-get install python3-dev libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev

# for videos
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev

# for GUI
sudo apt-get install libgtk-3-dev

# for optimization
sudo apt-get install libatlas-base-dev gfortran pylint

sudo apt-get install cmake-gui

sudo apt-get install gcc-7 g++-7
```

```bash
mkdir src/third_party && cd src/third_party
wget -O OpenCV-4.2.0.zip wget https://github.com/opencv/opencv/archive/4.2.0.zip 
unzip OpenCV-4.2.0.zip
rm OpenCV-4.2.0.zip
wget -O OpenCV_contrib-4.2.0.zip https://github.com/opencv/opencv_contrib/archive/4.2.0.zip
unzip OpenCV_contrib-4.2.0.zip
rm OpenCV_contrib-4.2.0.zip
```

From here, run the cmake-gui and set the source code to 
`/mnt/Data/cs5242-project/action-recognition/src/third_party/opencv-4.2.0` and the build directory to 
`/mnt/Data/cs5242-project/action-recognition/src/third_party/opencv-4.2.0/build` and click `Generate`. Then,  
set `PYTHON3_EXECUTABLE=/home/kennardng/anaconda3/envs/action-recognition/bin/python3`, 
`PYTHON3_INCLUDE_DIR=/home/kennardng/anaconda3/envs/action-recognition/include/python3.7m`
`PYTHON3_LIBRARY=/home/kennardng/anaconda3/envs/action-recognition/lib/libpython3.7m.so`
`PYTHON3_PACKAGES_PATH=/home/kennardng/anaconda3/envs/action-recognition/lib/python3.7/site-packages`
`OPENCV_EXTRA_MODULES=/mnt/Data/cs5242-project/action-recognition/src/third_party/opencv_contrib-4.2.0/modules`
`OPENCV_EXTRA_MODULES=/mnt/Data/cs5242-project/action-recognition/src/third_party/opencv_contrib-4.2.0/modules`
`CMAKE_INSTALL_PREFIX=/home/kennardng/anaconda3/envs/action-recognition/local`. 
`BUILD_opencv_xfeatures2d=0`
Don't build xfeatures_2d, we don't need it
Also, if your compiler is above gcc-8, then there will be some issues, I used `gcc-7` for installation and set
```bash
CUDA_HOST_COMPILER=/usr/bin/gcc-7
```
Remember to set numpy paths as well
Then, press `Configure` and `Generate`. Once the scripts are generated, go to `third_party/opencv-4.2.0/build` and 
use make 

There are some issues during `make install`, refer to [here](https://answers.opencv.org/question/221827/a-installation-problem-of-opencvsolved/)


```bash
CUDA_VISIBLE_DEVICES=0 python src/utils/optical_flow/extract_selflows.py --n_gpu 3 --gpu 0
CUDA_VISIBLE_DEVICES=1 python src/utils/optical_flow/extract_selflows.py --n_gpu 3 --gpu 1
CUDA_VISIBLE_DEVICES=3 python src/utils/optical_flow/extract_selflows.py --n_gpu 3 --gpu 2
```

## Extras




### Exporting Conda environments
```bash
# exporting the environment
conda env export > environment.yml

# installing the new environment.
conda env create -f environment.yml
```

