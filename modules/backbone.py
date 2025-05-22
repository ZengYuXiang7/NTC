# coding : utf-8
# Author : Yuxiang Zeng
import torch
from torch.nn import *

class Backbone(torch.nn.Module):
    def __init__(self, enc_in, config):
        super(Backbone, self).__init__()
        self.config = config
        self.rank = config.rank
        self.window = config.window
        self.user_embeds = Embedding(config.num_nodes, self.rank)
        self.item_embeds = Embedding(config.num_nodes, self.rank)
        self.time_embeds = Embedding(config.num_slots, self.rank)

        self.lstm = LSTM(self.rank, self.rank, batch_first=False)
        self.rainbow = torch.arange(-self.window + 1, 1).reshape(1, -1).to(config.device)
        self.attn = Sequential(Linear(2 * self.rank, 1), Tanh())
        self.user_linear = Linear(self.rank, self.rank)
        self.item_linear = Linear(self.rank, self.rank)
        self.time_linear = Linear(self.rank, self.rank)
        self.simple = False


    def to_seq_id(self, tids):
        tids = tids.reshape(-1, 1).repeat(1, self.window)
        tids += self.rainbow
        tids = tids.relu().permute(1, 0)
        return tids

    def forward(self, user, item, time, return_embeds=False):
        user_embeds = self.get_embeddings(user, "user")
        item_embeds = self.get_embeddings(item, "item")
        time_embeds = self.get_embeddings(time, "time")
        if not return_embeds:
            return self.get_score(user_embeds, item_embeds, time_embeds)
        else:
            return self.get_score(user_embeds, item_embeds, time_embeds), user_embeds, item_embeds, time_embeds

    def get_score(self, user_embeds, item_embeds, time_embeds):
        if self.simple:
            return self.get_final_score(user_embeds, item_embeds, time_embeds)
        # Interaction Modules
        user_embeds = self.user_linear(user_embeds)
        item_embeds = self.item_linear(item_embeds)
        time_embeds = self.time_linear(time_embeds)
        raw_score = torch.sum(user_embeds * item_embeds * time_embeds, dim=-1)
        pred = raw_score.sigmoid()
        return pred

    def get_embeddings(self, idx, select):
        if self.simple:
            return self.get_final_embeddings(idx, select)

        if select == "user":
            return self.user_embeds(idx)

        elif select == "item":
            return self.item_embeds(idx)

        elif select == "time":
            # Read Time Embeddings
            time_embeds = self.time_embeds(self.to_seq_id(idx))
            outputs, (hs, cs) = self.lstm.forward(time_embeds)
            # Attention [seq_len, batch, dim] -> [seq_len, batch, 1]
            hss = hs.repeat(self.window, 1, 1)
            attn = self.attn(torch.cat([outputs, hss], dim=-1))
            time_embeds = torch.sum(attn * outputs, dim=0)
            return time_embeds
        else:
            raise NotImplementedError("Unknown select type: {}".format(select))

    def get_final_embeddings(self, idx, select):
        if select == "user":
            user_embeds = self.user_embeds(idx)
            return self.user_linear(user_embeds)

        elif select == "item":
            item_embeds = self.item_embeds(idx)
            return self.item_linear(item_embeds)

        elif select == "time":
            # Read Time Embeddings
            time_embeds = self.time_embeds(self.to_seq_id(idx))
            outputs, (hs, cs) = self.lstm.forward(time_embeds)

            # Attention [seq_len, batch, dim] -> [seq_len, batch, 1]
            hss = hs.repeat(self.window, 1, 1)
            attn = self.attn(torch.cat([outputs, hss], dim=-1))
            time_embeds = torch.sum(attn * outputs, dim=0)
            time_embeds = self.time_linear(time_embeds)
            return time_embeds
        else:
            raise NotImplementedError("Unknown select type: {}".format(select))

    def get_final_score(self, user_embeds, item_embeds, time_embeds):
        pred = torch.sum(user_embeds * item_embeds * time_embeds, dim=-1).sigmoid()
        return pred
