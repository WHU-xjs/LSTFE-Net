from maskrcnn_benchmark.layers import Conv2d
from torchvision.ops import DeformConv2d
import torch
from torch import nn
from torch.nn import functional as F
import logging



class GetAlignedFeature(nn.Module):
	def __init__(self, in_dim):
		super(GetAlignedFeature, self).__init__()
		
		inter_dim = 2   # --------------------------- 1
		
		self.al_conv_one = nn.Conv2d(in_dim, inter_dim, kernel_size=3, stride=1, padding=1)
                #nn.init.kaiming_uniform_(self.conv1.weight, a=1)
		self.defConv1 = DeformConv2d(1024, 2048, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False)
		
		#self.al_conv_two = nn.Conv2d(2048, inter_dim, kernel_size=3, stride=1, padding=1)
		#self.defConv2 = DeformConv2d(2048, 2048, kernel_size=3, stride = 1, padding = 1, groups = 1, bias = False)
		
		#self.al_conv_three = nn.Conv2d(2048, inter_dim, kernel_size=3, stride=1, padding=1)
		#self.defConv3 = DeformConv2d(2048, 2048, kernel_size=3, stride = 1, padding = 1, groups = 1, bias = False)
		
		#self.al_conv_four = nn.Conv2d(2048, inter_dim, kernel_size=1, stride=1, padding=0)
		#self.defConv4 = DeformConv2d(1024, 2048, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False)

	def forward(self, refer_feat, sup_feat):
		concat_feat = torch.cat([refer_feat, sup_feat], dim=1)
                #logger.info("concat_feart: {}".format(concat_feat.shape))

		aligned_feat_1_offset = self.al_conv_one(concat_feat) # [1,18,38,50]
		#aligned_feat_1 = self.defConv1(concat_feat, aligned_feat_1_offset)
		aligned_feat = self.defConv1(sup_feat, aligned_feat_1_offset)
		
		#aligned_feat_2_offset = self.al_conv_two(aligned_feat_1)
		#aligned_feat_2 = self.defConv2(aligned_feat_1, aligned_feat_2_offset)
		
		#aligned_feat_3_offset = self.al_conv_three(aligned_feat_2)
		#aligned_feat_3 = self.defConv3(aligned_feat_2, aligned_feat_3_offset)
		
		#aligned_feat_4_offset = self.al_conv_four(aligned_feat_1)
		#aligned_feat = self.defConv4(sup_feat, aligned_feat_4_offset)   # 注意是sup_feat, 利用得到的偏移，对支持支持帧进行可变性卷积，以配准特征
		
		return aligned_feat
