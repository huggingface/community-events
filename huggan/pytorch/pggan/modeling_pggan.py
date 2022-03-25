import torch
import torch.nn as nn
import numpy as np
from utils import EqualizedLR_Conv2d, Pixel_norm, Minibatch_std

	
class FromRGB(nn.Module):
	def __init__(self, in_ch, out_ch):
		super().__init__()
		self.conv = EqualizedLR_Conv2d(in_ch, out_ch, kernel_size=(1, 1), stride=(1, 1))
		self.relu = nn.LeakyReLU(0.2)
		
	def forward(self, x):
		x = self.conv(x)
		return self.relu(x)

class ToRGB(nn.Module):
	def __init__(self, in_ch, out_ch):
		super().__init__()
		self.conv = EqualizedLR_Conv2d(in_ch, out_ch, kernel_size=(1,1), stride=(1, 1))
	
	def forward(self, x):

		return self.conv(x)


class G_Block(nn.Module):
	def __init__(self, in_ch, out_ch, initial_block=False):
		super().__init__()
		if initial_block:
			self.upsample = None
			self.conv1 = EqualizedLR_Conv2d(in_ch, out_ch, kernel_size=(4, 4), stride=(1, 1), padding=(3, 3))
		else:
			self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
			self.conv1 = EqualizedLR_Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.conv2 = EqualizedLR_Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.relu = nn.LeakyReLU(0.2)
		self.pixelwisenorm = Pixel_norm()
		nn.init.normal_(self.conv1.weight)
		nn.init.normal_(self.conv2.weight)
		nn.init.zeros_(self.conv1.bias)
		nn.init.zeros_(self.conv2.bias)

	def forward(self, x):

		if self.upsample is not None:
			x = self.upsample(x)
		# x = self.conv1(x*scale1)
		x = self.conv1(x)
		x = self.relu(x)
		x = self.pixelwisenorm(x)
		# x = self.conv2(x*scale2)
		x = self.conv2(x)
		x = self.relu(x)
		x = self.pixelwisenorm(x)
		return x


class D_Block(nn.Module):
	def __init__(self, in_ch, out_ch, initial_block=False):
		super().__init__()

		if initial_block:
			self.minibatchstd = Minibatch_std()
			self.conv1 = EqualizedLR_Conv2d(in_ch+1, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			self.conv2 = EqualizedLR_Conv2d(out_ch, out_ch, kernel_size=(4, 4), stride=(1, 1))
			self.outlayer = nn.Sequential(
									nn.Flatten(),
									nn.Linear(out_ch, 1)
									)
		else:			
			self.minibatchstd = None
			self.conv1 = EqualizedLR_Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			self.conv2 = EqualizedLR_Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
			self.outlayer = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

		self.relu = nn.LeakyReLU(0.2)
		nn.init.normal_(self.conv1.weight)
		nn.init.normal_(self.conv2.weight)
		nn.init.zeros_(self.conv1.bias)
		nn.init.zeros_(self.conv2.bias)
	
	def forward(self, x):
		if self.minibatchstd is not None:
			x = self.minibatchstd(x)
		
		x = self.conv1(x)
		x = self.relu(x)
		x = self.conv2(x)
		x = self.relu(x)
		x = self.outlayer(x)
		return x


class Generator(nn.Module):
	def __init__(self, latent_size, out_res):
		super().__init__()
		self.depth = 1
		self.alpha = 1
		self.fade_iters = 0
		self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
		self.current_net = nn.ModuleList([G_Block(latent_size, latent_size, initial_block=True)])
		self.toRGBs = nn.ModuleList([ToRGB(latent_size, 3)])
		# __add_layers(out_res)
		for d in range(2, int(np.log2(out_res))):
			if d < 6:
				## low res blocks 8x8, 16x16, 32x32 with 512 channels
				in_ch, out_ch = 512, 512
			else:
				## from 64x64(5th block), the number of channels halved for each block
				in_ch, out_ch = int(512 / 2**(d - 6)), int(512 / 2**(d - 5))
			self.current_net.append(G_Block(in_ch, out_ch))
			self.toRGBs.append(ToRGB(out_ch, 3))


	def forward(self, x):
		for block in self.current_net[:self.depth-1]:
			x = block(x)
		out = self.current_net[self.depth-1](x)
		x_rgb = self.toRGBs[self.depth-1](out)
		if self.alpha < 1:
			x_old = self.upsample(x)
			old_rgb = self.toRGBs[self.depth-2](x_old)
			x_rgb = (1-self.alpha)* old_rgb + self.alpha * x_rgb

			self.alpha += self.fade_iters

		return x_rgb
		

	def growing_net(self, num_iters):
		
		self.fade_iters = 1/num_iters
		self.alpha = 1/num_iters

		self.depth += 1


class Discriminator(nn.Module):
	def __init__(self, latent_size, out_res):
		super().__init__()
		self.depth = 1
		self.alpha = 1
		self.fade_iters = 0

		self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

		self.current_net = nn.ModuleList([D_Block(latent_size, latent_size, initial_block=True)])
		self.fromRGBs = nn.ModuleList([FromRGB(3, latent_size)])
		for d in range(2, int(np.log2(out_res))):
			if d < 6:
				in_ch, out_ch = 512, 512
			else:
				in_ch, out_ch = int(512 / 2**(d - 5)), int(512 / 2**(d - 6))
			self.current_net.append(D_Block(in_ch, out_ch))
			self.fromRGBs.append(FromRGB(3, in_ch))
	
	def forward(self, x_rgb):
		x = self.fromRGBs[self.depth-1](x_rgb)

		x = self.current_net[self.depth-1](x)
		if self.alpha < 1:

			x_rgb = self.downsample(x_rgb)
			x_old = self.fromRGBs[self.depth-2](x_rgb)
			x = (1-self.alpha)* x_old + self.alpha * x
			self.alpha += self.fade_iters
		for block in reversed(self.current_net[:self.depth-1]):
			x = block(x)

		return x
		
	def growing_net(self, num_iters):

		self.fade_iters = 1/num_iters
		self.alpha = 1/num_iters

		self.depth += 1