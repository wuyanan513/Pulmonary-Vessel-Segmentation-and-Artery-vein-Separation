#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F

smooth = 1.
epsilon = 1e-6


def sad_loss(pred, target, encoder_flag=True):
	"""
	AD: atention distillation loss
	: param pred: input prediction
	: param target: input target
	: param encoder_flag: boolean, True=encoder-side AD, False=decoder-side AD
	"""
	target = target.detach()
	if (target.size(-1) == pred.size(-1)) and (target.size(-2) == pred.size(-2)):
		# target and prediction have the same spatial resolution
		pass
	else:
		if encoder_flag == True:
			# target is smaller than prediction
			# use consecutive layers with scale factor = 2
			target = F.interpolate(target, scale_factor=2, mode='trilinear')
		else:
			# prediction is smaller than target
			# use consecutive layers with scale factor = 2
			pred = F.interpolate(pred, scale_factor=2, mode='trilinear')

	num_batch = pred.size(0)
	pred = pred.view(num_batch, -1)
	target = target.view(num_batch, -1)
	pred = F.softmax(pred, dim=1)
	target = F.softmax(target, dim=1)
	return F.mse_loss(pred, target)
def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5 : return [1,2,3]
    # Two dimensional
    elif len(shape) == 4 : return [1,2]
    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')
def Tversky_loss(pred, target):
    smooth = 1.0
    alpha = 0.05
    beta = 1 - alpha
    intersection = (pred*target).sum()
    FP = (pred*(1-target)).sum()
    FN = ((1-pred)*target).sum()
    return 1-(intersection + smooth)/(intersection + alpha*FP + beta*FN + smooth)
def asymmetric_focal_tversky_loss(y_true, y_pred):
	# Clip values to prevent division by zero error


	delta = 0.05
	gamma = 0.75
	epsilon = 1e-7

	y_pred = torch.clamp(y_pred, epsilon, 1. - epsilon)

	axis = identify_axis(y_true.size())
	# Calculate true positives (tp), false negatives (fn) and false positives (fp)
	tp = torch.sum(y_true * y_pred, axis=axis)
	fn = torch.sum(y_true * (1 - y_pred), axis=axis)
	fp = torch.sum((1 - y_true) * y_pred, axis=axis)
	dice_class = (tp + epsilon)/(tp + delta*fn + (1-delta)*fp + epsilon)

	#calculate losses separately for each class, only enhancing foreground class
	back_dice = (1-dice_class[:,0])
	fore_dice = (1-dice_class[:,1]) * torch.pow(1-dice_class[:,1], -gamma)

	# Average class scores
	loss = torch.mean(torch.stack([back_dice,fore_dice], axis=-1))
	return loss




def dice_loss(pred, target):
	"""
	DSC loss
	: param pred: input prediction
	: param target: input target
	"""
	iflat = pred.view(-1)
	tflat = target.view(-1)
	intersection = torch.sum((iflat * tflat))
	return 1. - ((2. * intersection + smooth)/(torch.sum(iflat) + torch.sum(tflat) + smooth))


def binary_cross_entropy(y_pred, y_true):
	"""
	Binary cross entropy loss
	: param y_pred: input prediction
	: param y_true: input target
	"""
	y_true = y_true.view(-1).float()
	y_pred = y_pred.view(-1).float()
	return F.binary_cross_entropy(y_pred, y_true)


def focal_loss(y_pred, y_true, alpha=0.25, gamma=2.0):
	"""
	Focal loss
	: param y_pred: input prediction
	: param y_true: input target
	: param alpha: balancing positive and negative samples, default=0.25
	: param gamma: penalizing wrong predictions, default=2
	"""
	# alpha balance weight for unbalanced positive and negative samples
	# clip to prevent NaN's and Inf's
	y_pred_flatten = torch.clamp(y_pred, min=epsilon, max=1. - epsilon)
	y_pred_flatten = y_pred_flatten.view(-1).float()
	y_true_flatten = y_true.detach()
	y_true_flatten = y_true_flatten.view(-1).float()
	loss = 0

	idcs = (y_true_flatten > 0)
	y_true_pos = y_true_flatten[idcs]
	y_pred_pos = y_pred_flatten[idcs]
	y_true_neg = y_true_flatten[~idcs]
	y_pred_neg = y_pred_flatten[~idcs]

	if y_pred_pos.size(0) != 0 and y_true_pos.size(0) != 0:
		# positive samples
		logpt = torch.log(y_pred_pos)
		loss += -1. * torch.mean(torch.pow((1. - y_pred_pos), gamma) * logpt) * alpha

	if y_pred_neg.size(0) != 0 and y_true_neg.size(0) != 0:
		# negative samples
		logpt2 = torch.log(1. - y_pred_neg)
		loss += -1. * torch.mean(torch.pow(y_pred_neg, gamma) * logpt2) * (1. - alpha)

	return loss


# def focal_loss(y_pred, y_true, alpha=0.75, gamma=2.0):
# 	"""
# 	Focal loss
# 	: param y_pred: input prediction
# 	: param y_true: input target
# 	: param alpha: balancing positive and negative samples, default=0.75
# 	: param gamma: penalizing wrong predictions, default=2
# 	"""
# 	# alpha balance weight for unbalanced positive and negative samples
# 	# clip to prevent NaN's and Inf's
# 	y_pred = F.relu(y_pred)
# 	y_pred = torch.clamp(y_pred, min=epsilon, max=1.-epsilon)
# 	y_true = y_true.view(-1).float()
# 	y_pred = y_pred.view(-1).float()
# 	idcs = (y_true > 0)
# 	y_true_pos = y_true[idcs]
# 	y_pred_pos = y_pred[idcs]
# 	y_true_neg = y_true[~idcs]
# 	y_pred_neg = y_pred[~idcs]
#
# 	if (y_pred_pos.size(0) != 0 and y_true_pos.size(0) != 0) and (y_pred_neg.size(0) != 0 and y_true_neg.size(0) != 0):
# 		# positive samples
# 		logpt = torch.log(y_pred_pos)
# 		loss = -1. * torch.mean(torch.pow((1. - y_pred_pos), gamma) * logpt) * alpha
# 		# negative samples
# 		logpt2 = torch.log(1. - y_pred_neg)
# 		loss += -1. * torch.mean(torch.pow(y_pred_neg, gamma) * logpt2) * (1. - alpha)
# 		return loss
# 	else:
# 		# use binary cross entropy to avoid NaN/Inf caused by missing positive or negative samples
# 		return F.binary_cross_entropy(y_pred, y_true)

