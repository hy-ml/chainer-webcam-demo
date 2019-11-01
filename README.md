# Webcam Sample
## Install
Assume that you use Anaconda.

Install requried packages by using pip.
```bash
conda create -n chainer-webcam-demo python=3.7
conda activate chainer-webcam-demo
CUDA_VERSION={cuda_version}
pip install chainer chainercv cuda-${CUDA_VERSION} opencv-python
```
If your cuda version is 9.2, CUDA_VERSION must be 92.

## Detection
Following models are supported.
- YOLOv2
- YOLOv3
- FasterRCNNFPNResNet50
- FasterRCNNFPNResNet101

GPU execution
```bash
python demo.py detection {model_name}
```

CPU execution
```bash
python demo.py detection {model_name} --gpu -1
```

Sample to demo YOLOv2 in CPU.
```bash
python demo.py detection YOLOv2 --gpu -1
```
