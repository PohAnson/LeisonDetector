import numpy as np
import os
import PIL.Image as Image

counter = 0
for i in os.listdir("HU_min"):
    counter += 1
    img = Image.open(f"HU_min/{i}")
    img.resize((512, 512))
    arr = np.asarray(img)

    print(arr.shape)
    np.save(f"tnpHU/{i[:-4]}", arr)
