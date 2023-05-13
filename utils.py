import sys
import os
import numpy as np
import torch
import torch.nn as nn
import SimpleITK as sitk
import pickle
from torch.nn.init import xavier_normal_, kaiming_normal_, constant_, normal_


smooth = 1.


def weights_init(net, init_type='normal'):
    """
    :param m: modules of CNNs
    :return: initialized modules
    """
    def init_func(m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            if init_type == 'normal':
                normal_(m.weight.data)
            elif init_type == 'xavier':
                xavier_normal_(m.weight.data)
            elif init_type == 'kaiming':
                kaiming_normal_(m.weight.data)
            if m.bias is not None:
                constant_(m.bias.data, 0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
    return


def load_pickle(filename='split_dataset.pickle'):
    """
    :param filename: pickle name
    :return: dictionary or list
    """
    with open(filename, 'rb') as handle:

        ids = pickle.load(handle)
    return ids


def save_pickle(dict, filename='split_dataset.pickle'):
    """
    :param dict: dictionary to be saved
    :param filename: pickle name
    :return: None
    """
    with open(filename, 'wb') as handle:
        pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return


def normalize_min_max(nparray):
    """
    :param nparray: input img (feature)
    :return: normalized nparray
    """
    nmin = np.amin(nparray)
    nmax = np.amax(nparray)
    norm_array = (nparray - nmin)/(nmax - nmin)
    return norm_array


def combine_total_avg(output, side_len, margin):
    """
    combine all things together and average overlapping areas of prediction
    curxinfo = [[curxdata, cursplitID, curnzhw, curshape, curorigin, curspacing]...]
    : param output: list of all coordinates and voxels of sub-volumes
    : param side_len: shape of the target volume
    : param margin: stride length of sliding window
    return: output_org, combined volume, original size
    return: curorigin, origin of CT
    return: curspacing, spacing of CT
    """
    curtemp = output[0]
    curshape = curtemp[3]
    curorigin = curtemp[4]
    curspacing = curtemp[5]
    #########################################################################
    nz,nh,nw = curtemp[2][0], curtemp[2][1], curtemp[2][2]
    [z, h, w] = curshape
    if type(margin) is not list:
        margin = [margin, margin, margin]

    splits = {}
    for i in range(len(output)):
        curinfo = output[i]
        curxdata = curinfo[0]
        cursplitID = int(curinfo[1])
        if not (cursplitID in splits.keys()):
            splits[cursplitID] = curxdata
        else:
            continue # only choose one splits

    output = np.zeros((z,h,w), np.float32)

    count_matrix = np.zeros((z,h,w), np.float32)

    idx = 0
    for iz in range(nz+1):
        for ih in range(nh+1):
            for iw in range(nw+1):
                sz = iz * side_len[0]
                ez = iz * side_len[0] + margin[0]
                sh = ih * side_len[1]
                eh = ih * side_len[1] + margin[1]
                sw = iw * side_len[2]
                ew = iw * side_len[2] + margin[2]
                if ez > z:
                    sz = z - margin[0]
                    ez = z
                if eh > h:
                    sh = h - margin[1]
                    eh = h
                if ew > w:
                    sw = w - margin[2]
                    ew = w
                split = splits[idx]
                ##assert (split.shape[0] == margin[0])
                ##assert (split.shape[1] == margin[1])
                ##assert (split.shape[2] == margin[2])
                #[margin[0]:margin[0] + side_len[0], margin[1]:margin[1] + \
                #side_len[1], margin[2]:margin[2] + side_len[2]]
                output[sz:ez, sh:eh, sw:ew] += split
                count_matrix[sz:ez, sh:eh, sw:ew] += 1
                idx += 1

    output = output/count_matrix
    output_org = output
    #output_org = output[:zorg, :horg, :worg]
    ##min_value = np.amin(output_org.flatten())
    ##max_value = np.amax(output_org.flatten())
    ##assert (min_value >= 0 and max_value <= 1)
    return output_org, curorigin, curspacing


def combine_total(output, side_len, margin):
    """
    combine all things together without average overlapping areas of prediction
    curxinfo = [[curxdata, cursplitID, curnzhw, curshape, curorigin, curspacing]...]
    : param output: list of all coordinates and voxels of sub-volumes
    : param side_len: shape of the target volume
    : param margin: stride length of sliding window
    return: output_org, combined volume, original size
    return: curorigin, origin of CT
    return: curspacing, spacing of CT
    """
    curtemp = output[0]
    curshape = curtemp[3]
    curorigin = curtemp[4]
    curspacing = curtemp[5]

    nz,nh,nw = curtemp[2][0], curtemp[2][1], curtemp[2][2]
    [z, h, w] = curshape
    #### output should be sorted
    if type(margin) is not list:
        margin = [margin, margin, margin]
    splits = {}
    for i in range(len(output)):
        curinfo = output[i]
        curxdata = curinfo[0]
        cursplitID = int(curinfo[1])
        splits[cursplitID] = curxdata

    output = -1000000 * np.ones((z,h,w), np.float32)

    idx = 0
    for iz in range(nz+1):
        for ih in range(nh+1):
            for iw in range(nw+1):
                sz = iz * side_len[0]
                ez = iz * side_len[0] + margin[0]
                sh = ih * side_len[1]
                eh = ih * side_len[1] + margin[1]
                sw = iw * side_len[2]
                ew = iw * side_len[2] + margin[2]
                if ez > z:
                    sz = z - margin[0]
                    ez = z
                if eh > h:
                    sh = h - margin[1]
                    eh = h
                if ew > w:
                    sw = w - margin[2]
                    ew = w
                split = splits[idx]
                ##assert (split.shape[0] == margin[0])
                ##assert (split.shape[1] == margin[1])
                ##assert (split.shape[2] == margin[2])
                output[sz:ez, sh:eh, sw:ew] = split
                idx += 1
    output_org = output
    #output_org = output[:z, :h, :w]
    ##min_value = np.amin(output_org.flatten())
    ##assert (min_value >= -1000000)
    return output_org, curorigin, curspacing


def save_itk(image, origin, spacing, filename):
    """
    :param image: images to be saved
    :param origin: CT origin
    :param spacing: CT spacing
    :param filename: save name
    :return: None
    """
    if type(origin) != tuple:
        if type(origin) == list:
            origin = tuple(reversed(origin))
        else:
            origin = tuple(reversed(origin.tolist()))
    if type(spacing) != tuple:
        if type(spacing) == list:
            spacing = tuple(reversed(spacing))
        else:
            spacing = tuple(reversed(spacing.tolist()))
    itkimage = sitk.GetImageFromArray(image, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    sitk.WriteImage(itkimage, filename, True)


def load_itk_image(filename):
    """
    :param filename: CT name to be loaded
    :return: CT image, CT origin, CT spacing
    """
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    return numpyImage, numpyOrigin, numpySpacing


def lumTrans(img):
    """
    :param img: CT image
    :return: Hounsfield Unit window clipped and normalized
    """
    lungwin = np.array([-1000.,400.])
    # the upper bound 400 is already ok
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')
    return newimg


class Logger(object):
    """
    Logger from screen to txt file
    """
    def __init__(self,logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass


def dice_coef_np(y_pred,y_true):
    """
    :param y_pred: prediction
    :param y_true: target ground-truth
    :return: dice coefficient
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def ppv_np(y_pred,y_true):
    """
    :param y_pred: prediction
    :param y_true: target ground-truth
    :return: positive predictive value
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (np.sum(y_pred_f) + smooth)


def sensitivity_np(y_pred,y_true):
    """
    :param y_pred: prediction
    :param y_true: target ground-truth
    :return: sensitivity
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (np.sum(y_true_f) + smooth)


def acc_np(y_pred,y_true):
    """
    :param y_pred: prediction
    :param y_true: target ground-truth
    :return: accuracy
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    y_pred_f = y_pred_f > 0.5
    intersection = np.sum(y_true_f==y_pred_f)
    return (intersection) / (len(y_true_f)+smooth)


def debug_dataloader(train_loader, testFolder):
    """
    :param train_loader: training data for debug
    :param testFolder: save directory
    :return: None
    """
    for i, (x, y, coord, org, spac, NameID, SplitID, nzhw, ShapeOrg) in enumerate(train_loader):
        xdata = x.numpy()
        ydata = y.numpy()
        origindata = org.numpy()
        spacingdata = spac.numpy()
        NameID = NameID[0]
        SplitID = SplitID[0]
        nzhw = nzhw.numpy()
        ShapeOrg = ShapeOrg.numpy()
        for j in range(xdata.shape[0]):
            if i < 2 and j < 5:
                curcube = xdata[j, 0]*255
                cury = ydata[j, 0]
                curorigin = origindata[j].tolist()
                curspacing = spacingdata[j].tolist()
                curnameID = NameID[j]
                print ('cursplit id ', SplitID[j])
                curpath = os.path.join(testFolder, 'test-%s-%d-%d-cube.nii.gz'%(curnameID, i, j))
                curypath = os.path.join(testFolder, 'test-%s-%d-%d-cubey.nii.gz'%(curnameID, i, j))
                curwpath = os.path.join(testFolder, 'test-%s-%d-%d-cubeweight.nii.gz'%(curnameID, i, j))
                save_itk(curcube.astype(dtype='uint8'), curorigin, curspacing, curpath)
                save_itk(cury.astype(dtype='uint8'), curorigin, curspacing, curypath)
    return


