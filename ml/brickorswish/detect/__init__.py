"""Detector + video tooling: video -> detections.json -> events.json + annotated video.

Only `crop` and `motion` are standard-library only; the rest import OpenCV, numpy and a detector lazily.
"""
