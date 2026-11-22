"""Detector + video tooling: video -> detections.json -> events.json + annotated video.

Only `crop` and `motion` are standard-library only. The rest import OpenCV, numpy and a detector lazily:
RF-DETR itself (PyTorch, Colab: ml/notebooks/detect_video.ipynb) or the same network exported to ONNX
(`onnx_model`, runs locally with only ml/requirements-local.txt).
"""
