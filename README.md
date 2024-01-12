# DSMN_dcase2023
![name](https://img.shields.io/badge/dsmn-v0.1.1-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)
![ipython](https://img.shields.io/badge/iPython-v8.4.0-orange)
![build](https://img.shields.io/badge/build-passing-yellowgreen)
## Low-Complexity Acoustic Scene Classification Using Deep Space Separable Distillation And Mutil-Task Learning 
[Paper Link](https://dcase.community/documents/challenge2023/technical_reports/DCASE2023_Wang_61_t1.pdf)
## Introduction 
This is a project that participated in Task 1 of DCASE2023  
   
|| Parms | MACs | Acc(float32) | Acc(int8) |
|----|----|----|----|----|
|model1| 45.1k | 8.64M | 53.4 | 53.3 |
|model2| 56.1k | 16.74M | 59.2 | 56.4 |
|model3| 56.5k | 25.44M | 53.4 | 50.8 |
|model4| 121.8k | 20.92M | 54.7 | 52.4 |

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
5、test
```
python test.py
```
## FAQ 
## Authors
Wang    eewkl@mail.scut.edu.cn  
Wu      202030242140@mail.scut.edu.cn

## Bibtex
If you think this work will help you, please refer to the following documents
```
@techreport{Wang2023,
    Author = "Wang, Kangli and Wu, Yiling and Li, Yanxiong",
    title = "Low-Complexity Acoustic Scene Classification Using Deep Space Separable Distillation Module and Multi-Label Learning",
    institution = "DCASE2023 Challenge",
    year = "2023",
    month = "May",
    abstract = "This technical report describes our system for Task 1 in Detection and Classification of Acoustic Scenes and Events (DCASE) 2023. We propose a deep space separable distillation block as the basic unit of the model, using its strong block processing ability to continuously cut the high-frequency and low-frequency parts of the log-Mel spectrogram. The accuracy is improved by multi-scale embedding and multi-task learning methods. To prevent overfitting, we adopt data augmentation methods such as mixing, speculation and spectral modulation. Quantization aware training is adopted to quantize the model to meet the requirements of edge devices with low complexity constraints. The proposed system achieves a 53.3\% accuracy on the development dataset with only a parameter count of 45.16 kB and the MACs of 8.64 M. "
}
```

## License
MIT
