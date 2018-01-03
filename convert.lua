require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'

----------------------------------------------------------

os.execute('git clone https://github.com/anuragranj/spynet')

----------------------------------------------------------

require('lfs').mkdir(('./models'))

for strFile in require('lfs').dir('./spynet/models') do
	if strFile:find('.t7') ~= nil then
		local moduleNetwork = torch.load('./spynet/models/' .. strFile)

		local intConv = 0
		for intLoad, moduleLoad in pairs(moduleNetwork:listModules()) do
			if torch.typename(moduleLoad) == 'cudnn.SpatialConvolution' then
				intConv = intConv + 1
				torch.save('./models/' .. strFile:gsub('.t7', '') .. '-' .. intConv .. '-weight.t7', moduleLoad.weight:float())
				torch.save('./models/' .. strFile:gsub('.t7', '') .. '-' .. intConv .. '-bias.t7', moduleLoad.bias:float())
			end
		end
	end
end