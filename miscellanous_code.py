
#### draw multiple figures with plt #####

import numpy as np
import matplotlib.pyplot as plt

w=10
h=10
size = 20
fig=plt.figure(figsize=(size, size))
columns = 4
rows = 5
for i in range(1, columns*rows +1):
    img = np.random.randint(10, size=(h,w))
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()



#### drawing patches


# img: img to draw the patches on
# patches: list of rectangle points to draw
def draw_patches(img, patches, format='ml'):
    import cv2
    new_img = np.copy(img)
    color = (0, 0, 255)
    if format == 'cv':
        if patches is not None:
            for i in range(len(patches)):
                cv2.rectangle(new_img, (int(patches[i][0]), int(patches[i][1])), \
                              (int(patches[i][0] + patches[i][2]), int(patches[i][1] + patches[i][3])), color, 2)

    if format == 'ml':
        if patches is not None:
            for i in range(len(patches)):
                cv2.rectangle(new_img, (int(patches[i][0]), int(patches[i][1])), \
                              (int(patches[i][2]), int(patches[i][3])), color, 2)

    return new_img
