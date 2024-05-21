import torch
import torch.nn as nn
from mmcv.cnn.bricks import ConvModule

class FeatureAggregator(nn.Module):
    def __init__(self,
                 num_convs=1,
                 channels=256,
                 kernel_size=3,
                 norm_cfg=None,
                 activation_cfg=dict(type='ReLU')):
        super(FeatureAggregator, self).__init__()
        assert num_convs > 0, 'The number of convs must be greater than 0.'

        self.embedding_convs = nn.ModuleList()
        for i in range(num_convs):
            if i == num_convs - 1:
                norm_cfg_final = None
                activation_cfg_final = None
            else:
                norm_cfg_final = norm_cfg
                activation_cfg_final = activation_cfg
            self.embedding_convs.append(
                ConvModule(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                    norm_cfg=norm_cfg_final,
                    act_cfg=activation_cfg_final))

        self.feature_fusions = nn.ModuleList()
        self.feature_fusions.append(
             ConvModule(
                 in_channels=channels * 8,
                 out_channels=channels * 4,
                 kernel_size=1,
                 padding=0,
                 norm_cfg=norm_cfg,
                 act_cfg=activation_cfg))
        self.feature_fusions.append(
             ConvModule(
                 in_channels=channels * 4,
                 out_channels=channels * 2,
                 kernel_size=3,
                 padding=1,
                 norm_cfg=norm_cfg,
                 act_cfg=activation_cfg))
        self.feature_fusions.append(
             ConvModule(
                 in_channels=channels * 2,
                 out_channels=channels,
                 kernel_size=1,
                 padding=0,
                 norm_cfg=None,
                 act_cfg=None))

    def forward(self, x, ref_x):
        """Aggregate reference feature maps `ref_x`.

        The aggregation mainly contains two steps:
        1. Computing the cosine similarity between `x` and `ref_x`.
        2. Use the normalized (i.e. softmax) cosine similarity to weightedly sum `ref_x`.

        Args:
            x (Tensor): of shape [1, C, H, W]
            ref_x (Tensor): of shape [N, C, H, W]. N is the number of reference feature maps.

        Returns:
            Tensor: The aggregated feature map with shape [1, C, H, W].
        """
        assert len(x.shape) == 4 and x.shape[0] == 1, "Only support 'batch_size == 1' for x"

        x_embedded = x
        for embed_conv in self.embedding_convs:
            x_embedded = embed_conv(x_embedded)
        x_embedded = x_embedded / x_embedded.norm(p=2, dim=1, keepdim=True)

        ref_x_embedded = ref_x
        for embed_conv in self.embedding_convs:
            ref_x_embedded = embed_conv(ref_x_embedded)
        ref_x_embedded = ref_x_embedded / ref_x_embedded.norm(p=2, dim=1, keepdim=True)

        fusion_input = torch.cat((x_embedded.repeat(ref_x_embedded.shape[0], 1, 1, 1),
                                  ref_x_embedded,
                                  x_embedded.repeat(ref_x_embedded.shape[0], 1, 1, 1) - ref_x_embedded,
                                  x.repeat(ref_x_embedded.shape[0], 1, 1, 1),
                                  ref_x,
                                  x.repeat(ref_x_embedded.shape[0], 1, 1, 1) - ref_x,
                                  -x_embedded.repeat(ref_x_embedded.shape[0], 1, 1, 1) + ref_x_embedded,
                                  -x.repeat(ref_x_embedded.shape[0], 1, 1, 1) + ref_x),
                                 dim=1)

        for feature_fusion in self.feature_fusions:
            fusion_input = feature_fusion(fusion_input)

        adaptive_weights = fusion_input
        adaptive_weights = adaptive_weights.softmax(dim=0)
        aggregated_feature_map = torch.sum(ref_x * adaptive_weights, dim=0, keepdim=True)

        return aggregated_feature_map
