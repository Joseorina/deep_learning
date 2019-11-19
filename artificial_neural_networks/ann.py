import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

dataset = pd.read_csv(abs_file_path)
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values