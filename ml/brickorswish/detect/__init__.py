"""Detector + video tooling: video -> detections.json -> events.json + annotated video.

Only `crop` is standard-library only. `video`, `annotate` and `colab` import OpenCV, RF-DETR (PyTorch)
or Colab modules lazily and are meant to run in the cloud (Colab), not on the dev laptop.
See ml/notebooks/detect_video.ipynb.
"""
