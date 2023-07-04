import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.nn import init
class DoubleAttention(nn.Module):

    def __init__(self, in_channels,c_m,c_n,reconstruct=True, res=False):
        super().__init__()
        self.res = res
        self.in_channels=in_channels
        self.reconstruct = reconstruct
        self.c_m=c_m
        self.c_n=c_n
        self.convA=nn.Conv3d(in_channels,c_m,kernel_size = 1)
        self.convB=nn.Conv3d(in_channels,c_n,kernel_size = 1)
        self.convV=nn.Conv3d(in_channels,c_n,kernel_size = 1)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv3d(c_m, in_channels, kernel_size = 1)
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w, d=x.shape
        assert c==self.in_channels
        A=self.convA(x) #b,c_m,h,w,d
        B=self.convB(x) #b,c_n,h,w,d
        V=self.convV(x) #b,c_n,h,w,d
        tmpA=A.view(b,self.c_m,-1)

        attention_maps=F.softmax(B.view(b,self.c_n,-1),dim=-1)
        attention_vectors=F.softmax(V.view(b,self.c_n,-1),dim=-1)
        # step 1: feature gating
        global_descriptors=torch.bmm(tmpA,attention_maps.permute(0,2,1)) #b.c_m,c_n
        # step 2: feature distribution
        tmpZ = global_descriptors.matmul(attention_vectors) #b,c_m,h*w
        tmpZ=tmpZ.view(b,self.c_m,h,w,d) #b,c_m,h,w
        if self.reconstruct:
            tmpZ=self.conv_reconstruct(tmpZ)
        if self.res:
            tmpZ = tmpZ+x
        return tmpZ 
class CoTAttention(nn.Module):

    def __init__(self, dim=512,kernel_size=3,res=False):
        super().__init__()
        self.res = res
        self.dim=dim
        self.kernel_size=kernel_size

        self.key_embed=nn.Sequential(
            nn.Conv3d(dim,dim,kernel_size=kernel_size,padding=kernel_size//2,groups=4,bias=False),
            nn.BatchNorm3d(dim),
            nn.ReLU()
        )
        self.value_embed=nn.Sequential(
            nn.Conv3d(dim,dim,1,bias=False),
            nn.BatchNorm3d(dim)
        )

        factor=4
        self.attention_embed=nn.Sequential(
            nn.Conv3d(2*dim,2*dim//factor,1,bias=False),
            nn.BatchNorm3d(2*dim//factor),
            nn.ReLU(),
            nn.Conv3d(2*dim//factor,kernel_size*kernel_size*dim,1)
        )


    def forward(self, x):
        bs,c,h,w,d=x.shape
        k1=self.key_embed(x) #bs,c,h,w
        v=self.value_embed(x).view(bs,c,-1) #bs,c,h,w

        y=torch.cat([k1,x],dim=1) #bs,2c,h,w
        att=self.attention_embed(y) #bs,c*k*k,h,w
        att=att.reshape(bs,c,self.kernel_size*self.kernel_size,h,w,d)
        att=att.mean(2,keepdim=False).view(bs,c,-1) #bs,c,h*w
        k2=F.softmax(att,dim=-1)*v
        k2=k2.view(bs,c,h,w,d)

        out = k1+k2


        return out
class UNet3D(nn.Module):
	"""
	Baseline model for pulmonary airway segmentation
	"""


	def __init__(self,in_channels=1, out_channels=1, coord=False
       ,Dmax=128, Hmax=128, Wmax=128):
		"""
		:param in_channels: input channel numbers
		:param out_channels: output channel numbers
		:param coord: boolean, True=Use coordinates as position information, False=not
		"""
		super(UNet3D, self).__init__()


		self._in_channels = in_channels
		self._out_channels = out_channels
		self._coord = coord
		self.dsconv6 = nn.Conv3d(128, 1, 3, 1, 1)  # deep supervision
		self.dsconv7 = nn.Conv3d(64, 1, 3, 1, 1)  # deep supervision
		self.dsconv8 = nn.Conv3d(32, 1, 3, 1, 1)  # deep supervision
		self.upsampling4 = nn.Upsample(scale_factor=4)
		self.upsampling8 = nn.Upsample(scale_factor=8)
		# self.pooling = nn.MaxPool3d(kernel_size=(2, 2, 2))
		# self.upsampling = nn.Upsample(scale_factor=2)
		self.pooling = nn.MaxPool3d(kernel_size=(2, 2, 2))
		self.upsampling = nn.Upsample(scale_factor=2)
		self.conv1 = nn.Sequential(
			nn.Conv3d(in_channels=self._in_channels, out_channels=8, kernel_size=3, stride=1, padding=1),
			nn.InstanceNorm3d(8),
			nn.ReLU(inplace=True),
			nn.Conv3d(8, 16, 3, 1, 1),
			nn.InstanceNorm3d(16),
			nn.ReLU(inplace=True))
		# self.conv1x1 = nn.Sequential(
		# 	nn.Conv3d(in_channels=self._in_channels, out_channels=16, kernel_size=1, stride=1),
		# 	nn.InstanceNorm3d(16),
		# 	nn.ReLU(inplace=True),)
		self.conv2 = nn.Sequential(
			CoTAttention(16,3),
			nn.InstanceNorm3d(16),
			nn.ReLU(inplace=True),
			nn.Conv3d(16, 32, 3, 1, 1),
			nn.InstanceNorm3d(32),
			nn.ReLU(inplace=True))
		# self.conv2x1 = nn.Sequential(
		# 	nn.Conv3d(in_channels=16, out_channels=32, kernel_size=1, stride=1),
		# 	nn.InstanceNorm3d(32),
		# 	nn.ReLU(inplace=True), )

		self.conv3 = nn.Sequential(
			CoTAttention(32,3),
			nn.InstanceNorm3d(32),
			nn.ReLU(inplace=True),
			nn.Conv3d(32, 64, 3, 1, 1),
			nn.InstanceNorm3d(64),
			nn.ReLU(inplace=True))
		# self.conv3x1 = nn.Sequential(
		# 	nn.Conv3d(in_channels=32, out_channels=64, kernel_size=1, stride=1),
		# 	nn.InstanceNorm3d(64),
		# 	nn.ReLU(inplace=True), )
	
		self.conv4 = nn.Sequential(
			CoTAttention(64,3),
			nn.InstanceNorm3d(64),
			nn.ReLU(inplace=True),
			nn.Conv3d(64, 128, 3, 1, 1),
			nn.InstanceNorm3d(128),
			nn.ReLU(inplace=True))
		# self.conv4x1 = nn.Sequential(
		# 	nn.Conv3d(in_channels=64, out_channels=128, kernel_size=1, stride=1),
		# 	nn.InstanceNorm3d(128),
		# 	nn.ReLU(inplace=True), )

		self.conv5 = nn.Sequential(
			CoTAttention(128, 3),
			nn.InstanceNorm3d(128),
			nn.ReLU(inplace=True),
			nn.Conv3d(128, 256, 3, 1, 1),
			nn.InstanceNorm3d(256),
			nn.ReLU(inplace=True))
		# self.conv5x1 = nn.Sequential(
		# 	nn.Conv3d(in_channels=128, out_channels=256, kernel_size=1, stride=1),
		# 	nn.InstanceNorm3d(256),
		# 	nn.ReLU(inplace=True), )

		self.conv6 = nn.Sequential(
			nn.Conv3d(256 + 128, 128, kernel_size=3, stride=1, padding=1),
			nn.InstanceNorm3d(128),
			nn.ReLU(inplace=True),
			CoTAttention(128,3),
			nn.InstanceNorm3d(128),
			nn.ReLU(inplace=True))
		# self.conv6x1 = nn.Sequential(
		# 	nn.Conv3d(in_channels=384, out_channels=128, kernel_size=1, stride=1),
		# 	nn.InstanceNorm3d(128),
		# 	nn.ReLU(inplace=True), )

		self.conv7 = nn.Sequential(
			nn.Conv3d(64 + 128, 64, 3, 1, 1),
			nn.InstanceNorm3d(64),
			nn.ReLU(inplace=True),
			CoTAttention(64,3),
			nn.InstanceNorm3d(64),
			nn.ReLU(inplace=True))
		# self.conv7x1 = nn.Sequential(
		# 	nn.Conv3d(in_channels=128 + 64, out_channels=64, kernel_size=1, stride=1),
		# 	nn.InstanceNorm3d(64),
		# 	nn.ReLU(inplace=True), )

		self.conv8 = nn.Sequential(
			nn.Conv3d(32 + 64, 32, 3, 1, 1),
			nn.InstanceNorm3d(32),
			nn.ReLU(inplace=True),
			CoTAttention(32,3),
			nn.InstanceNorm3d(32),
			nn.ReLU(inplace=True))
		# self.conv8x1 = nn.Sequential(
		# 	nn.Conv3d(in_channels=64 + 32, out_channels=32, kernel_size=1, stride=1),
		# 	nn.InstanceNorm3d(32),
		# 	nn.ReLU(inplace=True), )
		
		if self._coord:
			num_channel_coord = 3
		else:
			num_channel_coord = 0
		self.conv9 = nn.Sequential(
			nn.Conv3d(16 + 32 + num_channel_coord, 16, 3, 1, 1),
			nn.InstanceNorm3d(16),
			nn.ReLU(inplace=True),
			CoTAttention(16,3),
			nn.InstanceNorm3d(16),
			nn.ReLU(inplace=True))
		# self.conv9x1 = nn.Sequential(
		# 	nn.Conv3d(in_channels=32 + 16, out_channels=16, kernel_size=1, stride=1),
		# 	nn.InstanceNorm3d(16),
		# 	nn.ReLU(inplace=True), )
		self.a_skip1 = DoubleAttention(16,32,32,True)
		self.a_skip2 = DoubleAttention(32,64,64,True)
		self.a_skip3 = DoubleAttention(64,128,128,True)
		self.a_skip4 = DoubleAttention(128,256,256,True)
		self.sigmoid = nn.Sigmoid()
		self.conv10 = nn.Conv3d(16, self._out_channels, 1, 1, 0)

	def forward(self, input, coordmap=None):
		"""
		:param input: shape = (batch_size, num_channels, D, H, W) \
		:param coordmap: shape = (batch_size, 3, D, H, W)
		:return: output segmentation tensor, attention mapping
		"""
		conv1 = self.conv1(input)
		a_conv1 = self.a_skip1(conv1)

		x = self.pooling(conv1)

		conv2 = self.conv2(x)
		a_conv2 = self.a_skip2(conv2)
		x = self.pooling(conv2)

		conv3 = self.conv3(x)
		a_conv3 = self.a_skip3(conv3)
		x = self.pooling(conv3)

		conv4 = self.conv4(x)
		a_conv4 = self.a_skip4(conv4)
		x = self.pooling(conv4)


		conv5 = self.conv5 (x)
		# res5 = self.conv5x1(x)
		# conv5 = conv5 + res5



		x = self.upsampling(conv5)
		x = torch.cat([x, a_conv4], dim=1)
		conv6 = self.conv6(x)
		# res6 = self.conv6x1(x)
		# conv6= conv6 + res6




		ds_6 = self.sigmoid(self.upsampling8(self.dsconv6(conv6)))

		x = self.upsampling(conv6)
		x = torch.cat([x, a_conv3], dim=1)
		conv7 = self.conv7(x)
		# res7 = self.conv7x1(x)
		# conv7 = conv7 + res7
		ds_7=self.sigmoid(self.upsampling4(self.dsconv7(conv7)))

		x = self.upsampling(conv7)
		x = torch.cat([x, a_conv2], dim=1)
		conv8 = self.conv8(x)

		ds_8 = self.sigmoid(self.upsampling(self.dsconv8(conv8)))



		x = self.upsampling(conv8)

		if (self._coord is True) and (coordmap is not None):
			x = torch.cat([x, a_conv1, coordmap], dim=1)
		else:
			x = torch.cat([x, a_conv1], dim=1)

		conv9 = self.conv9(x)
		# res9= self.conv9x1(x)
		# conv9 = conv9 + res9



		x = self.conv10(conv9)
		x = self.sigmoid(x)

		mapping3 = torch.sum(torch.pow(conv3, exponent=2), dim=1, keepdim=True)
		mapping4 = torch.sum(torch.pow(conv4, exponent=2), dim=1, keepdim=True)
		mapping5 = torch.sum(torch.pow(conv5, exponent=2), dim=1, keepdim=True)
		mapping6 = torch.sum(torch.pow(conv6, exponent=2), dim=1, keepdim=True)
		mapping7 = torch.sum(torch.pow(conv7, exponent=2), dim=1, keepdim=True)
		mapping8 = torch.sum(torch.pow(conv8, exponent=2), dim=1, keepdim=True)
		mapping9 = torch.sum(torch.pow(conv9, exponent=2), dim=1, keepdim=True)

		return [x,ds_6,ds_7,ds_8], [mapping3, mapping4, mapping5, mapping6, mapping7, mapping8, mapping9]

		# conv5 = self.conv5(x)





if __name__ == '__main__':
	os.environ['CUDA_VISIBLE_DEVICES']='1'
	net = UNet3D(
        in_channels=1, out_channels=1)
	net = net.cuda()
	x = torch.rand(2, 1, 96, 128, 192)
	x = x.cuda()
	y = net(x)

	print(net)
	print(y[0])
	print('Number of network parameters:', sum(param.numel() for param in net.parameters()))
# Number of network parameters: 4118849 Baseline

