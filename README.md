# Webcam Sample
## Detection
Following models are supported.
- YOLOv2
- YOLOv3
- FasterRCNNFPNResNet50
- FasterRCNNFPNResNet101

GPU execution
```python
python demo.py detection {model_name}
```

CPU execution
```python
python demo.py detection {model_name} --gpu -1
```

Sample to demo YOLOv2 in CPU.
```python
python demo.py detection YOLOv2 --gpu -1
```
