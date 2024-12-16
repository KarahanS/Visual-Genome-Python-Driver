# read sample2.txt and get 300 images
with open("sample2.txt", "r") as f:
    images = f.readlines()

images = [img.strip() for img in images]
print(images[:5])

import random
import numpy as np
from collections import defaultdict

# select 300 images randomly
images = random.sample(images, 200)

# print to subsample.txt
with open("subsample.txt", "w") as f:
    for img in images:
        f.write(f"{img}\n")


# Generate comparisons...
