# coding : utf-8
# Author : Yuxiang Zeng
# 每次开展新实验都改一下这里
from layers.metric.distance import PairwiseLoss
from exp.exp_base import BasicModel
from modules.backbone import Backbone

class Model(BasicModel):
    def __init__(self, datamodule, config):
        super().__init__(config)
        self.config = config
        # self.input_size = datamodule.train_loader.dataset..shape[-1]
        self.input_size = 3
        self.hidden_size = config.rank

        if config.model == 'ours':
            self.model = Backbone(self.input_size, config)

        else:
            raise ValueError(f"Unsupported model type: {config.model}")

