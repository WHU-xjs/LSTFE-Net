# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""
import time

from PIL import Image
from collections import deque

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads

from .deform_conv import GetAlignedFeature
#from mega_core.utils.logger import setup_logger
from .local_feat_aggregator import FeatureAggregator

class GeneralizedRCNNLSTFE(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNNLSTFE, self).__init__()
        self.device = cfg.MODEL.DEVICE

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

        self.global_enable = cfg.MODEL.VID.LSTFE.GLOBAL.ENABLE
        self.frame_selector_enable = cfg.MODEL.VID.LSTFE.SELECT.ENABLE
        self.align_enable = cfg.MODEL.VID.LSTFE.ALIGN.ENABLE

        self.global_select_num = cfg.MODEL.VID.LSTFE.REF_NUM_GLOBAL

        self.base_num = cfg.MODEL.VID.RPN.REF_POST_NMS_TOP_N
        self.advanced_num = int(self.base_num * cfg.MODEL.VID.LSTFE.RATIO)

        self.all_frame_interval = cfg.MODEL.VID.LSTFE.ALL_FRAME_INTERVAL
        self.key_frame_location = cfg.MODEL.VID.LSTFE.KEY_FRAME_LOCATION

        self.local_feat_aggregator = FeatureAggregator(num_convs=2, channels=1024)
        self.get_aligned_local = GetAlignedFeature(1024*3)

    def forward(self, images, targets=None):
        """
        Arguments:
            #images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            images["cur"] = to_image_list(images["cur"])
            images["ref_l"] = [to_image_list(image) for image in images["ref_l"]]
            images["ref_g"] = [to_image_list(image) for image in images["ref_g"]]

            return self._forward_train(images["cur"], images["ref_l"], images["ref_g"], targets)
        else:
            images["cur"] = to_image_list(images["cur"])
            images["ref_l"] = [to_image_list(image) for image in images["ref_l"]]
            images["ref_g"] = [to_image_list(image) for image in images["ref_g"]]

            infos = images.copy()
            infos.pop("cur")
            return self._forward_test(images["cur"], infos)

    def _forward_train(self, img_cur, imgs_l, imgs_g, targets):

        # build local frames
        concat_imgs_l = torch.cat([img_cur.tensors, *[img.tensors for img in imgs_l]], dim=0)
        concat_feats_l, c5_local_feats = self.backbone(concat_imgs_l)

        num_imgs = 1 + len(imgs_l)
        feats_l_list = torch.chunk(concat_feats_l, num_imgs, dim=0)

        c5_feats_l_list = torch.chunk(c5_local_feats, num_imgs, dim=0)

        # build global frames
        proposals_g_list = []
        if imgs_g:
            concat_imgs_g = torch.cat([img.tensors for img in imgs_g], dim=0)
            concat_feats_g, c5_global_feats = self.backbone(concat_imgs_g)
            feats_g_list = torch.chunk(concat_feats_g, len(imgs_g), dim=0)
            c5_feats_g_list = torch.chunk(c5_global_feats, len(imgs_g), dim=0)

            if self.frame_selector_enable:
                cur_feat = c5_feats_l_list[0]
                def get_global_feat(cur_feat, global_feats, num_select):
                    from torch.nn.functional import adaptive_avg_pool2d
                    
                    avg_pool_cur = adaptive_avg_pool2d(cur_feat, (1,1))
                    avg_pool_cur_squeeze = avg_pool_cur.squeeze() #[2048]
                    
                    sim_all = []
                    for feat in global_feats:
                        avg_pool_feat = adaptive_avg_pool2d(feat, (1,1))
                        feat_squeeze = avg_pool_feat.squeeze()  # 压缩
                        sim = torch.cosine_similarity(avg_pool_cur_squeeze, feat_squeeze, dim=0)
                        sim_all.append(sim)
                    sorted_indices = sorted(range(len(sim_all)), key=lambda i: sim_all[i])
                    chosen_ids = sorted_indices[:num_select]
                    return chosen_ids
                ids = get_global_feat(cur_feat, c5_feats_g_list[0:], num_select = self.global_select_num)
                index = [i for i in ids]
            else:
                index = [0, 1]

            feats_g_list = [feats_g_list[i] for i in index]
            c5_feats_g_list = [c5_feats_g_list[i] for i in index]
            for i in range(len(index)):
                proposals_ref = self.rpn(imgs_g[i], (feats_g_list[i], ), version="ref")
                proposals_g_list.append(proposals_ref[0])
        else:
            feats_g_list = []

        if self.align_enable:
            feat_cur = feats_l_list[0]
            feat_local = torch.cat([feats_l_list[1], feats_l_list[2]], dim = 1)
            feat_local_res = self.get_aligned_local(feat_cur, feat_local)
            feat_local_res_list = torch.chunk(feat_local_res, 2, dim = 1)
            # aggreate local feats into cur_feat
            feat_cur = self.local_feat_aggregator(feat_cur, torch.cat([feat_local_res_list[0], feat_local_res_list[1]], dim = 0)) 
            feat_local_new = (feat_cur, feat_local_res_list[0], feat_local_res_list[1])
            feats_l_list = feat_local_new 

        proposals, proposal_losses = self.rpn(img_cur, (feats_l_list[0],), targets, version="key")

        proposals_l_list = []
        proposals_cur = self.rpn(img_cur, (feats_l_list[0], ), version="ref")
        proposals_l_list.append(proposals_cur[0])
        for i in range(len(imgs_l)):
            proposals_ref = self.rpn(imgs_l[i], (feats_l_list[i + 1], ), version="ref")
            proposals_l_list.append(proposals_ref[0])


        feats_list = [feats_l_list, feats_g_list]
        proposals_list = [proposals, proposals_l_list, proposals_g_list]
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(feats_list,
                                                        proposals_list,
                                                        targets)
        else:
            detector_losses = {}

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def _forward_test(self, imgs, infos, targets=None):
        """
        forward for the test phase.
        :param imgs:
        :param infos:
        :param targets:
        :return:
        """
        def update_feature(img=None, feats=None, proposals=None, proposals_feat=None):
            assert (img is not None) or (feats is not None and proposals is not None and proposals_feat is not None)

            if img is not None:
                feats = self.backbone(img)[0]

                if self.align_enable:
                    feat_cur = feats
                    concat_imgs_l = torch.cat([img.tensors for img in infos['ref_l']], dim=0)
                    feats_local = self.backbone(concat_imgs_l)[0]
                    feats_local = torch.cat([feats_local, feats_local], dim=1)
                    feat_local_res = self.get_aligned_local(feat_cur, feats_local)
                    feat_local_res_list = torch.chunk(feat_local_res, 2, dim = 1)
                    feat_local_res_list_concat = torch.cat([feat_local_res_list[0], feat_local_res_list[1]], dim = 0)
                    feat_cur_res = self.local_feat_aggregator(feat_cur, feat_local_res_list_concat)
                    feats = feat_cur_res

                # note here it is `imgs`! for we only need its shape, it would not cause error, but is not explicit.
                proposals = self.rpn(imgs, (feats,), version="ref")
                proposals_feat = self.roi_heads.box.feature_extractor(feats, proposals, pre_calculate=True)

            self.feats.append(feats)
            self.proposals.append(proposals[0])
            self.proposals_dis.append(proposals[0][:self.advanced_num])
            self.proposals_feat.append(proposals_feat)
            self.proposals_feat_dis.append(proposals_feat[:self.advanced_num])

        if targets is not None:
            raise ValueError("In testing mode, targets should be None")

        if infos["frame_category"] == 0:  # a new video
            self.seg_len = infos["seg_len"]
            self.end_id = 0

            self.feats = deque(maxlen=self.all_frame_interval)
            self.proposals = deque(maxlen=self.all_frame_interval)
            self.proposals_dis = deque(maxlen=self.all_frame_interval)
            self.proposals_feat = deque(maxlen=self.all_frame_interval)
            self.proposals_feat_dis = deque(maxlen=self.all_frame_interval)

            # self.roi_heads.box.feature_extractor.init_memory()
            if self.global_enable:
                self.roi_heads.box.feature_extractor.init_global()

            feats_cur = self.backbone(imgs.tensors)[0]
            proposals_cur = self.rpn(imgs, (feats_cur, ), version="ref")
            proposals_feat_cur = self.roi_heads.box.feature_extractor(feats_cur, proposals_cur, pre_calculate=True)


            while len(self.feats) < self.key_frame_location + 1:
                update_feature(None, feats_cur, proposals_cur, proposals_feat_cur)

            while len(self.feats) < self.all_frame_interval:
                self.end_id = min(self.end_id + 1, self.seg_len - 1)
                end_filename = infos["pattern"] % self.end_id
                end_image = Image.open(infos["img_dir"] % end_filename).convert("RGB")

                end_image = infos["transforms"](end_image)
                if isinstance(end_image, tuple):
                    end_image = end_image[0]
                end_image = end_image.view(1, *end_image.shape).to(self.device)

                update_feature(end_image)

        elif infos["frame_category"] == 1:
            self.end_id = min(self.end_id + 1, self.seg_len - 1)
            end_image = infos["ref_l"][0].tensors

            update_feature(end_image)

        # 1. update global
        if infos["ref_g"]:
            for global_img in infos["ref_g"]:
                feats = self.backbone(global_img.tensors)[0]
                proposals = self.rpn(global_img, (feats,), version="ref")
                proposals_feat = self.roi_heads.box.feature_extractor(feats, proposals, pre_calculate=True)

                self.roi_heads.box.feature_extractor.update_global(proposals_feat)

        feats = self.feats[self.key_frame_location]
        proposals, proposal_losses = self.rpn(imgs, (feats, ), None)

        proposals_ref = cat_boxlist(list(self.proposals))
        proposals_ref_dis = cat_boxlist(list(self.proposals_dis))
        proposals_feat_ref = torch.cat(list(self.proposals_feat), dim=0)
        proposals_feat_ref_dis = torch.cat(list(self.proposals_feat_dis), dim=0)

        proposals_list = [proposals, proposals_ref, proposals_ref_dis, proposals_feat_ref, proposals_feat_ref_dis]

        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(feats, proposals_list, None)
        else:
            result = proposals

        return result
