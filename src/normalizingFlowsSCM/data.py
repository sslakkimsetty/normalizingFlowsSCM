
import numpy as np
import pandas as pd

CARTOON_PATH = "/content/drive/MyDrive/data/cartoon-faces/"
cartoon_features_images_fname = "cartoon_features_filenames.csv"

cartoon_features_images = pd.read_csv(CARTOON_PATH+cartoon_features_images_fname)
cartoon_features_images = cartoon_features_images[["face_shape", "glasses", "filename"]]

cartoon_features_images.glasses.value_counts()
