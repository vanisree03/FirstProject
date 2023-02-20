import pandas as pd
import numpy as np
import os
import shutil
import glob
import matplotlib.pyplot as plt

ROOT_DIR = "kaggle dataset/COVID-19_Radiography_Dataset/"
imgs = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

NEW_DIR = "all_images/"

if not os.path.exists(NEW_DIR):
    os.mkdir(NEW_DIR)

    for i in imgs:
        org_dir = os.path.join(ROOT_DIR, i + "/")

        for imgfile in glob.iglob(os.path.join(org_dir, "*.png")):
            shutil.copy(imgfile, NEW_DIR)

else:
    print("Already Exist")

counter = {'COVID': 0, 'Lung_Opacity': 0, 'Normal': 0, 'Viral Pneumonia': 0}

for image in imgs:
    for count in glob.iglob(NEW_DIR + image + "*"):
        counter[image] += 1

print(counter)

plt.figure(figsize=(10, 5))
plt.bar(x=counter.keys(), height=counter.values())
plt.show()

# Creating the folder

if not os.path.exists(NEW_DIR + "train_test_split/"):

    os.makedirs(NEW_DIR + "train_test_split/")

    os.makedirs(NEW_DIR + "train_test_split/train/Normal")
    os.makedirs(NEW_DIR + "train_test_split/train/Covid")

    os.makedirs(NEW_DIR + "train_test_split/test/Normal")
    os.makedirs(NEW_DIR + "train_test_split/test/Covid")

    os.makedirs(NEW_DIR + "train_test_split/validation/Normal")
    os.makedirs(NEW_DIR + "train_test_split/validation/Covid")

    # Train Data
    for i in np.random.choice(replace=False, size=3000, a=glob.glob(NEW_DIR + imgs[0] + "*")):
        shutil.copy(i, NEW_DIR + "train_test_split/train/Covid")
        os.remove(i)

    for i in np.random.choice(replace=False, size=3900, a=glob.glob(NEW_DIR + imgs[2] + "*")):
        shutil.copy(i, NEW_DIR + "train_test_split/train/Normal")
        os.remove(i)

    for i in np.random.choice(replace=False, size=900, a=glob.glob(NEW_DIR + imgs[3] + "*")):
        shutil.copy(i, NEW_DIR + "train_test_split/train/Covid")
        os.remove(i)

    # Validation Data
    for i in np.random.choice(replace=False, size=308, a=glob.glob(NEW_DIR + imgs[0] + "*")):
        shutil.copy(i, NEW_DIR + "train_test_split/validation/Covid")
        os.remove(i)

    for i in np.random.choice(replace=False, size=500, a=glob.glob(NEW_DIR + imgs[2] + "*")):
        shutil.copy(i, NEW_DIR + "train_test_split/validation/Normal")
        os.remove(i)

    for i in np.random.choice(replace=False, size=200, a=glob.glob(NEW_DIR + imgs[3] + "*")):
        shutil.copy(i, NEW_DIR + "train_test_split/validation/Covid")
        os.remove(i)

    # Test Data
    for i in np.random.choice(replace=False, size=300, a=glob.glob(NEW_DIR + imgs[0] + "*")):
        shutil.copy(i, NEW_DIR + "train_test_split/test/Covid")
        os.remove(i)

    for i in np.random.choice(replace=False, size=300, a=glob.glob(NEW_DIR + imgs[2] + "*")):
        shutil.copy(i, NEW_DIR + "train_test_split/test/Normal")
        os.remove(i)

    for i in np.random.choice(replace=False, size=200, a=glob.glob(NEW_DIR + imgs[3] + "*")):
        shutil.copy(i, NEW_DIR + "train_test_split/test/Covid")
        os.remove(i)

train_path = "all_images/train_test_split/train"
valid_path = "all_images/train_test_split/validation"
test_path = "all_images/train_test_split/test"
