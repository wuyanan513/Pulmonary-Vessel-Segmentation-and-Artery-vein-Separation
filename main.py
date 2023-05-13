import os
import time
import numpy as np
import data
from importlib import import_module
import shutil
from trainval_classifier import train_casenet, val_casenet
from utils import Logger, save_itk, weights_init, debug_dataloader
import sys
sys.path.append('../')
from split_combine_mj import SplitComb
import torch
from torch.nn import DataParallel
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
import csv
from option import parser


def main():
	global args
	LR = 0.003
	args = parser.parse_args()
	torch.manual_seed(42)
	print('----------------------Load Model------------------------')
	model = import_module(args.model)
	config, net = model.get_model(args)
	start_epoch = args.start_epoch
	save_dir = args.save_dir
	save_dir = os.path.join('results',save_dir)
	print("savedir: ", save_dir)
	print("args.lr: ", args.lr)
	args.lr_stage = config['lr_stage']
	args.lr_preset = config['lr']
	# checkpoint = torch.load('/mnt/data-hdd/wuyanan/airway/code/SSL/result/017.ckpt')
	# net.load_state_dict(checkpoint['state_dict'])

	weights_init(net, init_type='kaiming')  # weight initialization


	if args.epochs is None:
		end_epoch = args.lr_stage[-1]
	else:
		end_epoch = args.epochs
		
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	logfile = os.path.join(save_dir, 'log.txt')
	
	if args.test != 1:
		sys.stdout = Logger(logfile)
		pyfiles = [f for f in os.listdir('./') if f.endswith('.py')]
		for f in pyfiles:
			shutil.copy(f, os.path.join(save_dir, f))

	net = net.cuda()
	cudnn.benchmark = True
	if args.multigpu:
		net = DataParallel(net)

	if args.cubesizev is not None:
		marginv = args.cubesizev
	else:
		marginv = args.cubesize
	print('validation stride ', args.stridev)


	optimizer = optim.Adam(net.parameters(), lr=LR)

	if args.test:
		print('---------------------testing---------------------')
		split_comber = SplitComb(args.stridev, marginv)
		dataset_test = data.AirwayData(
			config,
			phase='test',
			split_comber=split_comber,
			debug=args.debug,
			random_select=False)
		test_loader = DataLoader(
			dataset_test,
			batch_size=args.batch_size,
			shuffle=False,
			num_workers=args.workers,
			pin_memory=True)
		epoch = 1
		print('start testing')
		testdata = val_casenet(epoch, net, test_loader, args, save_dir, test_flag=True)
		return

	if args.debugval:
		epoch = 1
		print ('---------------------validation---------------------')
		split_comber = SplitComb(args.stridev, marginv)
		dataset_val = data.AirwayData(
			config,
			phase='val',
			split_comber=split_comber,
			debug=args.debug,
			random_select=False)
		val_loader = DataLoader(
			dataset_val,
			batch_size=args.batch_size,
			shuffle=False,
			num_workers=args.workers,
			pin_memory=True)
		valdata = val_casenet(epoch, net, val_loader, args, save_dir)
		return

	print('---------------------------------Load Dataset--------------------------------')
	margin = args.cubesize
	print('patch size ', margin)
	print('train stride ', args.stridet)
	split_comber = SplitComb(args.stridet, margin)

	dataset_train = data.AirwayData(
		config,
		phase='train',
		split_comber=split_comber,
		debug=args.debug,
		random_select=args.randsel)

	train_loader = DataLoader(
		dataset_train,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.workers,
		pin_memory=True,
	drop_last=True)
	
	print('--------------------------------------')
	split_comber = SplitComb(args.stridev, marginv)

	# load validation dataset
	dataset_val = data.AirwayData(
		config,
		phase='val',
		split_comber=split_comber,
		debug=args.debug,
		random_select=False)
	val_loader = DataLoader(
		dataset_val,
		batch_size=1,
		shuffle=False,
		num_workers=args.workers,
		pin_memory=True)

	print('--------------------------------------')

	# load testing dataset
	# dataset_test = data.AirwayData(
	# 	config,
	# 	phase='test',
	# 	split_comber=split_comber,
	# 	debug=args.debug,
	# 	random_select=False)
	# test_loader = DataLoader(
	# 	dataset_test,
	# 	batch_size=args.batch_size,
	# 	shuffle=False,
	# 	num_workers=args.workers,
	# 	pin_memory=True)

	# if args.debugdataloader and args.debug:
	# 	print ('start debugging')
	# 	testFolder = 'debug'
	# 	if not os.path.exists(testFolder):
	# 		os.mkdir(testFolder)
	# 	debug_dataloader(train_loader, testFolder)
	# 	return

	##############################
	# start training
	##############################
	
	total_epoch = []
	train_loss = []
	val_loss = []
	test_loss = []

	train_acc = []
	val_acc = []
	test_acc = []

	train_sensi = []
	val_sensi = []
	test_sensi = []

	dice_train = []
	dice_val = []
	dice_test = []

	ppv_train=[]
	ppv_val=[]
	ppv_test = []
	
	logdirpath = os.path.join(save_dir, 'log')
	if not os.path.exists(logdirpath):
		os.mkdir(logdirpath)

	v_loss, mean_acc2, mean_sensiti2, mean_dice2, mean_ppv2 = 0, 0, 0, 0, 0
	te_loss, mean_acc3, mean_sensiti3, mean_dice3, mean_ppv3 = 0, 0, 0, 0, 0
	scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
	for epoch in range(start_epoch, end_epoch + 1):
		t_loss, mean_acc, mean_sensiti, mean_dice, mean_ppv = train_casenet(epoch, net, train_loader, optimizer, args, save_dir)
		train_loss.append(t_loss)
		train_acc.append(mean_acc)
		train_sensi.append(mean_sensiti)
		dice_train.append(mean_dice)
		ppv_train.append(mean_ppv)

		# Save the current model
		if args.multigpu:
			state_dict = net.module.state_dict()
		else:
			state_dict = net.state_dict()
		for key in state_dict.keys():
			state_dict[key] = state_dict[key].cpu()
		torch.save({
			'state_dict': state_dict,
			'args': args},
			os.path.join(save_dir, 'latest.ckpt'))
		
		# Save the model frequently
		if epoch % args.save_freq == 0:            
			if args.multigpu:
				state_dict = net.module.state_dict()
			else:
				state_dict = net.state_dict()
			for key in state_dict.keys():
				state_dict[key] = state_dict[key].cpu()
			torch.save({
				'state_dict': state_dict,
				'args': args},
				os.path.join(save_dir, '%03d.ckpt' % epoch))

		if (epoch % args.val_freq == 0) or (epoch == start_epoch):
			v_loss, mean_acc2, mean_sensiti2, mean_dice2, mean_ppv2 = val_casenet(epoch, net, val_loader, args, save_dir)

		# if epoch % args.test_freq == 0:
		# 	te_loss, mean_acc3, mean_sensiti3, mean_dice3, mean_ppv3 = val_casenet(epoch, net, test_loader, args, save_dir, test_flag=True)
		
		val_loss.append(v_loss)
		val_acc.append(mean_acc2)
		val_sensi.append(mean_sensiti2)
		dice_val.append(mean_dice2)
		ppv_val.append(mean_ppv2)
		scheduler.step()

		test_loss.append(te_loss)
		test_acc.append(mean_acc3)
		test_sensi.append(mean_sensiti3)
		dice_test.append(mean_dice3)
		ppv_test.append(mean_ppv3)
		
		total_epoch.append(epoch)

		totalinfo = np.array([total_epoch, train_loss, val_loss, test_loss, train_acc, val_acc, test_acc,
							  train_sensi, val_sensi, test_sensi, dice_train, dice_val, dice_test,
							  ppv_train, ppv_val, ppv_test])
		np.save(os.path.join(logdirpath, 'log.npy'), totalinfo)

	logName = os.path.join(logdirpath, 'log.csv')
	with open(logName, 'a') as csvout:
		writer = csv.writer(csvout)
		row = ['train epoch', 'train loss', 'val loss', 'test loss', 'train acc', 'val acc', 'test acc',
			   'train sensi', 'val sensi', 'test sensi', 'dice train', 'dice val', 'dice test',
			   'ppv train','ppv val', 'ppv test']
		writer.writerow(row)

		for i in range(len(total_epoch)):
			row = [total_epoch[i], train_loss[i], val_loss[i], test_loss[i],
				   train_acc[i], val_acc[i], test_acc[i],
				   train_sensi[i], val_sensi[i], test_sensi[i],
				   dice_train[i], dice_val[i], dice_test[i],
				   ppv_train[i], ppv_val[i], ppv_test[i]]
			writer.writerow(row)
		csvout.close()

	print("Done")
	return


if __name__ == '__main__':
	os.environ['CUDA_VISIBLE_DEVICES']='1'
	main()

