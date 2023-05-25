# DSMN_dcase2023
![name](https://img.shields.io/badge/dsmn-v0.1.1-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)
![ipython](https://img.shields.io/badge/iPython-v8.4.0-orange)
![build](https://img.shields.io/badge/build-passing-yellowgreen)

## Introduction 
This is a project that participated in Task 1 of DCASE2023  

Waiting for update  
our Technical Report: https://dcase.community/challenge2023/task-low-complexity-acoustic-scene-classification

## Requirements
The required library files are placed in requirements.txt  
our environment: RTX3080 + cuda11.3 + torch1.11
## Usage
1、Clone project
```
git clone https://github.com/Wangkkklll/DSMN_dcase2023.git
```
2、Install the required library files
```
pip install -r requirements
```
3、Dataset  
Please place dataset under the path /DSMN_dcase2023/data  
If your data is in another path, please create soft link
```
ln -s /your_path/ /root/DSMN_dcase2023/data
```
4、How to train
```
python train.py
```
## FAQ 
## Authors
Wang    eewkl@mail.scut.edu.cn  
Wu      202030242140@mail.scut.edu.cn
## License
MIT
