import numpy as np
from PIL import Image

def ious(pred, target, num_class):
    ious = np.zeros((num_class))
    for cls in range(num_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious[cls]=0
            #ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            #ious.append(float(intersection) / max(union, 1))
            ious[cls]=float(intersection) / max(union, 1)
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return np.mean(ious)

def pixel_accs(pred, target):
    correct = (pred == target).sum()
    total   = (target == target).sum()
    return correct / total

def dice_coefs(pred, target, num_class):
    smooth = 1e-5
    dice_coefs = np.zeros((num_class))
    for cls in range(num_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        dice_coefa=(2 * intersection + smooth) / (pred_inds.sum() + target_inds.sum() + smooth)
        dice_coefs[cls]=float(dice_coefa)
    return np.mean(dice_coefs)

def save_result_comparison(epoch, input_np, input_mask, output_np):
    original_im = np.zeros((512,512))
    original_im[:,:]=input_np[0,0,:,:]
    im_seg = np.zeros((512, 512))
    im_mask = np.zeros((512, 512))

    # the following version is designed for 11-class version and could still work if the number of classes is fewer.
    for i in range(512):
        for j in range(512):
            if output_np[i, j] == 0:
                im_seg[i, j] = 255
            elif output_np[i, j] == 1:
                im_seg[i, j] = 0
            if input_mask[0,i,j] == 0:
                im_mask[i, j] = 255
            elif input_mask[0,i,j] == 1:
                im_mask[i, j] = 0

    # horizontally stack original image and its corresponding segmentation results
    hstack_image = np.hstack((original_im, im_seg, im_mask))
    new_im = Image.fromarray(np.uint8(hstack_image))
    file_name = '/home/yqw/seg/20201030/0.95,0.99/output/' + str(epoch) + '.jpg'
    new_im.save(file_name)
    print('successfully save result')
