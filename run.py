#!/usr/bin/env python2.7

import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys
import torch
import torch.utils.serialization

##########################################################

arguments_strModel = 'sintel-final'
arguments_strFirst = './images/first.png'
arguments_strSecond = './images/second.png'
arguments_strOut = './result.flo'

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
	if strOption == '--model':
		arguments_strModel = strArgument # which model to use, see below

	elif strOption == '--first':
		arguments_strFirst = strArgument # path to the first frame

	elif strOption == '--second':
		arguments_strSecond = strArgument # path to the second frame

	elif strOption == '--out':
		arguments_strOut = strArgument # path to where the output should be stored

	# end
# end

if arguments_strModel == 'chairs-clean':
	arguments_strModel = '4'
	
elif arguments_strModel == 'chairs-final':
	arguments_strModel = '3'
	
elif arguments_strModel == 'sintel-clean':
	arguments_strModel = 'C'
	
elif arguments_strModel == 'sintel-final':
	arguments_strModel = 'F'
	
elif arguments_strModel == 'kitti-final':
	arguments_strModel = 'K'

# end

##########################################################

class Network(torch.nn.Module):
	def __init__(self):
		super(Network, self).__init__()

		class Preprocess(torch.nn.Module):
			def __init__(self):
				super(Preprocess, self).__init__()
			# end

			def forward(self, variableInput):
				variableBlue = variableInput[:, 0:1, :, :] - 0.406
				variableGreen = variableInput[:, 1:2, :, :] - 0.456
				variableRed = variableInput[:, 2:3, :, :] - 0.485

				variableBlue = variableBlue / 0.225
				variableGreen = variableGreen / 0.224
				variableRed = variableRed / 0.229

				return torch.cat([variableRed, variableGreen, variableBlue], 1)
			# end
		# end

		class Basic(torch.nn.Module):
			def __init__(self, intLevel):
				super(Basic, self).__init__()

				self.moduleBasic = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
				)

				if intLevel == 5:
					if arguments_strModel == '3' or arguments_strModel == '4':
						intLevel = 4 # the models trained on the flying chairs dataset do not come with weights for the sixth layer
					# end
				# end

				for intConv in range(5):
					self.moduleBasic[intConv * 2].weight.data.copy_(torch.utils.serialization.load_lua('./models/modelL' + str(intLevel + 1) + '_' + arguments_strModel  + '-' + str(intConv + 1) + '-weight.t7'))
					self.moduleBasic[intConv * 2].bias.data.copy_(torch.utils.serialization.load_lua('./models/modelL' + str(intLevel + 1) + '_' + arguments_strModel  + '-' + str(intConv + 1) + '-bias.t7'))
				# end
			# end

			def forward(self, variableInput):
				return self.moduleBasic(variableInput)
			# end
		# end

		class Backward(torch.nn.Module):
			def __init__(self):
				super(Backward, self).__init__()
			# end

			def forward(self, variableInput, variableFlow):
				if hasattr(self, 'tensorGrid') == False or self.tensorGrid.size(0) != variableInput.size(0) or self.tensorGrid.size(2) != variableInput.size(2) or self.tensorGrid.size(3) != variableInput.size(3):
					torchHorizontal = torch.linspace(-1.0, 1.0, variableInput.size(3)).view(1, 1, 1, variableInput.size(3)).expand(variableInput.size(0), 1, variableInput.size(2), variableInput.size(3))
					torchVertical = torch.linspace(-1.0, 1.0, variableInput.size(2)).view(1, 1, variableInput.size(2), 1).expand(variableInput.size(0), 1, variableInput.size(2), variableInput.size(3))

					self.tensorGrid = torch.cat([ torchHorizontal, torchVertical ], 1).cuda()
				# end

				variableGrid = torch.autograd.Variable(data=self.tensorGrid, volatile=not self.training)

				variableFlow = torch.cat([ variableFlow[:, 0:1, :, :] / ((variableInput.size(3) - 1.0) / 2.0), variableFlow[:, 1:2, :, :] / ((variableInput.size(2) - 1.0) / 2.0) ], 1)

				return torch.nn.functional.grid_sample(input=variableInput, grid=(variableGrid + variableFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')
			# end
		# end

		self.modulePreprocess = Preprocess()

		self.moduleBasic = torch.nn.ModuleList([ Basic(intLevel) for intLevel in range(6) ])

		self.moduleBackward = Backward()
	# end

	def forward(self, variableFirst, variableSecond):
		variableFlow = []

		variableFirst = [ self.modulePreprocess(variableFirst) ]
		variableSecond = [ self.modulePreprocess(variableSecond) ]

		for intLevel in range(5):
			if variableFirst[0].size(2) > 32 or variableFirst[0].size(3) > 32:
				variableFirst.insert(0, torch.nn.functional.avg_pool2d(input=variableFirst[0], kernel_size=2, stride=2))
				variableSecond.insert(0, torch.nn.functional.avg_pool2d(input=variableSecond[0], kernel_size=2, stride=2))
			# end
		# end

		variableFlow = torch.autograd.Variable(data=torch.zeros(variableFirst[0].size(0), 2, int(math.floor(variableFirst[0].size(2) / 2.0)), int(math.floor(variableFirst[0].size(3) / 2.0))).cuda(), volatile=not self.training)

		for intLevel in range(len(variableFirst)):
			variableUpsampled = torch.nn.functional.upsample(input=variableFlow, scale_factor=2, mode='bilinear') * 2.0

			if variableUpsampled.size(2) != variableFirst[intLevel].size(2): variableUpsampled = torch.nn.functional.pad(input=variableUpsampled, pad=[0, 0, 0, 1], mode='replicate')
			if variableUpsampled.size(3) != variableFirst[intLevel].size(3): variableUpsampled = torch.nn.functional.pad(input=variableUpsampled, pad=[0, 1, 0, 0], mode='replicate')

			variableFlow = self.moduleBasic[intLevel](torch.cat([ variableFirst[intLevel], self.moduleBackward(variableSecond[intLevel], variableUpsampled), variableUpsampled ], 1)) + variableUpsampled
		# end

		return variableFlow
	# end
# end

moduleNetwork = Network().cuda()

##########################################################

def estimate(tensorInputFirst, tensorInputSecond):
	tensorOutput = torch.FloatTensor()

	assert(tensorInputFirst.size(1) == tensorInputSecond.size(1))
	assert(tensorInputFirst.size(2) == tensorInputSecond.size(2))

	intWidth = tensorInputFirst.size(2)
	intHeight = tensorInputFirst.size(1)

	assert(intWidth == 640) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
	assert(intHeight == 480) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

	if True:
		tensorInputFirst = tensorInputFirst.cuda()
		tensorInputSecond = tensorInputSecond.cuda()
		tensorOutput = tensorOutput.cuda()
	# end

	if True:
		variableInputFirst = torch.autograd.Variable(data=tensorInputFirst.view(1, 3, intHeight, intWidth), volatile=True)
		variableInputSecond = torch.autograd.Variable(data=tensorInputSecond.view(1, 3, intHeight, intWidth), volatile=True)

		tensorOutput.resize_(2, intHeight, intWidth).copy_(moduleNetwork(variableInputFirst, variableInputSecond).data[0])
	# end

	if True:
		tensorInputFirst = tensorInputFirst.cpu()
		tensorInputSecond = tensorInputSecond.cpu()
		tensorOutput = tensorOutput.cpu()
	# end

	return tensorOutput
# end

##########################################################

if __name__ == '__main__':
	tensorInputFirst = torch.FloatTensor(numpy.rollaxis(numpy.asarray(PIL.Image.open(arguments_strFirst))[:, :, ::-1], 2, 0).astype(numpy.float32) / 255.0)
	tensorInputSecond = torch.FloatTensor(numpy.rollaxis(numpy.asarray(PIL.Image.open(arguments_strSecond))[:, :, ::-1], 2, 0).astype(numpy.float32) / 255.0)

	tensorOutput = estimate(tensorInputFirst, tensorInputSecond)

	objectOutput = open(arguments_strOut, 'wb')

	numpy.array([80, 73, 69, 72], numpy.uint8).tofile(objectOutput)
	numpy.array([tensorOutput.size(2), tensorOutput.size(1)], numpy.int32).tofile(objectOutput)

	for intY in range(tensorOutput.size(1)):
		for intX in range(tensorOutput.size(2)):
			numpy.array([ tensorOutput[0, intY, intX], tensorOutput[1, intY, intX] ], numpy.float32).tofile(objectOutput)
		# end
	# end

	objectOutput.close()
# end