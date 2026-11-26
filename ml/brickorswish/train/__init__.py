"""Training our own detector (cloud only: ml/notebooks/train_detector.ipynb).

`drafts` and `dataset` are standard-library only (drafts reads the video with OpenCV); `fit` needs rfdetr (PyTorch).
"""

# The order is the model's class index (0, 1, 2, 3): the exported model.json maps indices back to these names.
CLASSES = ("ball", "ball_in_net", "rim", "person")
