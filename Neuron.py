from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
A=np.load('/home/amax/data/yqw/neuron/images.npy')
label=np.load('/home/amax/data/yqw/neuron/labels.npy')
def histequ(gray, nlevels=256):
    # Compute histogram
    histogram = np.bincount(gray.flatten(), minlength=nlevels)
    #print ("histogram: ", histogram)

    # Mapping function
    uniform_hist = (nlevels - 1) * (np.cumsum(histogram)/(gray.size * 1.0))
    uniform_hist = uniform_hist.astype('uint8')
    #print ("uniform hist: ", uniform_hist)

    # Set the intensity of the pixel in the raw gray to its corresponding new intensity
    height, width = gray.shape
    uniform_gray = np.zeros(gray.shape, dtype='uint8')  # Note the type of elements
    for i in range(height):
        for j in range(width):
            uniform_gray[i,j] = uniform_hist[gray[i,j]]

    return uniform_gray
#preprocessing
A=A.astype(np.float)
for a in range(A.shape[0]):
    im = A[a,:,:]
    im=(im-np.min(im))/(np.max(im)-np.min(im))
    #print(im)
    im=im*255
    #print(im)
    im=im.astype(np.uint8)
    im = histequ(im)
    A[a, :, :] = im
#cut
A=A[:,4:260,4:260]
label=label[:,4:260,4:260]
print(A.shape,label.shape)
np.save('/home/yqw/neuron/image.npy',A)
np.save('/home/yqw/neuron/label.npy',label)
#check plots
im=A[20,:,:]
input_mask=label[20,:,:]
im_mask = np.zeros((264, 264))
for i in range(264):
    for j in range(264):
        if input_mask[i, j] == 0:
            im_mask[i, j] = 255
        elif input_mask[i, j] == 1:
           im_mask[i, j] = 0
hstack_image = np.hstack((im, im_mask))
image = Image.fromarray(hstack_image)
image=image.convert('L')
image.save('/home/yqw/neuron/check/outfile.png')



