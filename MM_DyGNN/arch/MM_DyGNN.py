import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sipbuild.generator.parser.tokens import keywords
from triton.ops import attention

from .graph import gcn
from .crossspareatt import CrossAttentionLayerTopK


class DMSTGCN(nn.Module):
    def __init__(self,num_nodes, dropout=0.3,
                 out_dim=12, residual_channels=32 ,dilation_channels=32, end_channels=512,
                 kernel_size=2, blocks=4, layers=2, days=144, dims=32, order=2, in_dim=1,
                 attention_dim=128,feed_forward_dim=512,num_heads=4,normalization="batch",**kwargs):
        super(DMSTGCN, self).__init__()
        # device = kwargs.get('device', 'cuda:0')  # Default to 'cpu' if device is not provided
        # Time of Day Embedding
        self.steps_per_day = days  # Default to 288 time steps per day (5-minute intervals)
        self.days_per_week = 7
        self.tod_embedding_dim = residual_channels # Match residual_channels
        self.dow_embedding_dim =  residual_channels# Match residual_channels

        # Time embeddings
        self.tod_embedding = nn.Parameter(torch.empty(self.steps_per_day, self.tod_embedding_dim))
        self.dow_embedding = nn.Parameter(torch.empty(self.days_per_week, self.dow_embedding_dim))
        nn.init.xavier_uniform_(self.tod_embedding)
        nn.init.xavier_uniform_(self.dow_embedding)
        skip_channels = 8
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.days = days
        # tcn layer
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        # self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.normal = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.filter_convs_a = nn.ModuleList()
        self.gate_convs_a = nn.ModuleList()
        # self.residual_convs_a = nn.ModuleList()
        self.skip_convs_a = nn.ModuleList()
        self.normal_a = nn.ModuleList()
        self.gconv_a = nn.ModuleList()

        self.filter_convs_b = nn.ModuleList()
        self.gate_convs_b = nn.ModuleList()
        # self.residual_convs_b = nn.ModuleList()
        self.skip_convs_b = nn.ModuleList()
        self.normal_b = nn.ModuleList()
        self.gconv_b = nn.ModuleList()

        self.normal_graph= nn.ModuleList()
        self.normal_graph_a = nn.ModuleList()
        self.normal_graph_b = nn.ModuleList()
        # attention layer
        self.attenLayer = nn.ModuleList()
        self.attenLayer_a = nn.ModuleList()
        self.attenLayer_b = nn.ModuleList()

        self.gconv_a2p = nn.ModuleList()

        # start conv
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))

        self.start_conv_a = nn.Conv2d(in_channels=in_dim,
                                      out_channels=residual_channels,
                                      kernel_size=(1, 1))
        self.start_conv_b = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))


        receptive_field = 1
        #
        self.supports_len = 1
        self.nodevec_p1 = nn.Parameter(torch.randn(days, dims), requires_grad=True) # 288, 40,40
        self.nodevec_p2 = nn.Parameter(torch.randn(num_nodes, dims), requires_grad=True) # 491, 40
        self.nodevec_p3 = nn.Parameter(torch.randn(num_nodes, dims), requires_grad=True)  # 491, 40
        self.nodevec_pk = nn.Parameter(torch.randn(dims, dims, dims), requires_grad=True) # 40, 40, 40
        #
        self.nodevec_a1 = nn.Parameter(torch.randn(days, dims), requires_grad=True)
        self.nodevec_a2 = nn.Parameter(torch.randn(num_nodes, dims), requires_grad=True)
        self.nodevec_a3 = nn.Parameter(torch.randn(num_nodes, dims), requires_grad=True)
        self.nodevec_ak = nn.Parameter(torch.randn(dims, dims, dims), requires_grad=True)
        #
        self.nodevec_bp1 = nn.Parameter(torch.randn(days, dims), requires_grad=True)
        self.nodevec_bp2 = nn.Parameter(torch.randn(num_nodes, dims), requires_grad=True)
        self.nodevec_bp3 = nn.Parameter(torch.randn(num_nodes, dims), requires_grad=True)
        self.nodevec_bpk = nn.Parameter(torch.randn(dims, dims, dims), requires_grad=True)

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):

                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                # add tcn for a part
                self.filter_convs_a.append(nn.Conv2d(in_channels=residual_channels,
                                                     out_channels=dilation_channels,
                                                     kernel_size=(1, kernel_size), dilation=new_dilation))
    
                self.gate_convs_a.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))


  

                self.skip_convs_a.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))

                self.filter_convs_b.append(nn.Conv2d(in_channels=residual_channels,
                                                     out_channels=dilation_channels,
                                                     kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs_b.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))




                self.skip_convs_b.append(nn.Conv2d(in_channels=dilation_channels,
                                                   out_channels=skip_channels,
                                                   kernel_size=(1, 1)))

                if normalization == "batch":
                    self.normal.append(nn.BatchNorm2d(residual_channels))
                    self.normal_a.append(nn.BatchNorm2d(residual_channels))
                    self.normal_b.append(nn.BatchNorm2d(residual_channels))
                    self.normal_graph.append(nn.BatchNorm2d(residual_channels))
                    self.normal_graph_a.append(nn.BatchNorm2d(residual_channels))
                    self.normal_graph_b.append(nn.BatchNorm2d(residual_channels))
                elif normalization == "layer":
                    self.normal.append(nn.LayerNorm([residual_channels, num_nodes, 13 - receptive_field - new_dilation + 1]))
                    self.normal_a.append(nn.LayerNorm([residual_channels, num_nodes, 13 - receptive_field - new_dilation + 1]))
                    self.normal_b.append(nn.LayerNorm([residual_channels, num_nodes, 13 - receptive_field - new_dilation + 1]))

                    self.normal_graph.append(
                        nn.LayerNorm([residual_channels, num_nodes, 13 - receptive_field - new_dilation + 1]))
                    self.normal_graph_a.append(
                        nn.LayerNorm([residual_channels, num_nodes, 13 - receptive_field - new_dilation + 1]))
                    self.normal_graph_b.append(
                        nn.LayerNorm([residual_channels, num_nodes, 13 - receptive_field - new_dilation + 1]))

                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                self.gconv.append(
                    gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len, order=order))
                self.gconv_a.append(
                    gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len, order=order))
                self.gconv_b.append(
                    gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len, order=order))
                self.attenLayer.append(CrossAttentionLayerTopK(dilation_channels,attention_dim,feed_forward_dim,num_heads,dropout,kwargs))
                self.attenLayer_a.append(CrossAttentionLayerTopK(dilation_channels, attention_dim, feed_forward_dim, num_heads, dropout,kwargs))
                self.attenLayer_b.append(CrossAttentionLayerTopK(dilation_channels, attention_dim, feed_forward_dim, num_heads, dropout,kwargs))
             
        self.relu = nn.ReLU(inplace=True)

        # output layer for primary a and b
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels * (12 + 10 + 9 + 7 + 6 + 4 + 3 + 1),
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_1_a = nn.Conv2d(in_channels=skip_channels * (12 + 10 + 9 + 7 + 6 + 4 + 3 + 1),
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2_a = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_1_b = nn.Conv2d(in_channels=skip_channels * (12 + 10 + 9 + 7 + 6 + 4 + 3 + 1),
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2_b = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)


        self.receptive_field = receptive_field
        self.log_sigma = nn.Parameter(torch.zeros(3))





    def dgconstruct(self, time_embedding, source_embedding, target_embedding, core_embedding):
        adp = torch.einsum('ai, ijk->ajk', time_embedding, core_embedding) # 288, 40, 40
        adp = torch.einsum('bj, ajk->abk', source_embedding, adp) # 491, 491, 40
        adp = torch.einsum('ck, abk->abc', target_embedding, adp) # 491,491,
        adp = F.softmax(F.relu(adp), dim=2)
        return adp

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int,**kerwargs):
        """Feed forward
        Args:
            history_data (torch.Tensor): history data. [B, L, N, C].
            future_data (torch.Tensor): future data. [B, L, N, C].
            train (bool): is training or not.
        """
        """
        input: (B, C, N, L)
        """



        inputs = history_data.permute(0, 3, 2, 1)
        ind = (history_data[:,0,0,3]*self.days % self.days).int()
        in_len = inputs.size(3)
        if in_len < self.receptive_field:
            xo = nn.functional.pad(inputs, (self.receptive_field - in_len, 0, 0, 0))
        else:
            xo = inputs
        x = self.start_conv(xo[:, [0]])#[64, 32, 491, 13]
        x_a = self.start_conv_a(xo[:, [1]])
        x_b = self.start_conv_b(xo[:, [2]])
        # Extract temporal indices from the last time step for each node (STID approach)
        tod_data = history_data[:, -1, :, 3]  # [B, N]
        dow_data = history_data[:, -1, :, 4]  # [B, N]
        tod_idx = (tod_data * self.steps_per_day).long() % self.steps_per_day
        dow_idx = (dow_data * self.days_per_week).long() % self.days_per_week

        # Get embeddings
        tod_emb = self.tod_embedding[tod_idx]  # [B, N, residual_channels]
        dow_emb = self.dow_embedding[dow_idx]  # [B, N, residual_channels]
        # Reshape embeddings to match convolution output dimensions
        batch_size, channels, num_nodes, seq_len = x.shape
        tod_emb = tod_emb.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, seq_len)
        dow_emb = dow_emb.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, seq_len)

        # Add temporal embeddings to enrich the features
        x = x + tod_emb + dow_emb
        x_a = x_a + tod_emb + dow_emb
        x_b = x_b + tod_emb + dow_emb

        skip = 0
        skip_a = 0
        skip_b = 0

        # dynamic graph construction ind=[64][000000...0]
        adp = self.dgconstruct(self.nodevec_p1[ind], self.nodevec_p2, self.nodevec_p3, self.nodevec_pk)#[64, 491, 491]
        adp_a = self.dgconstruct(self.nodevec_a1[ind], self.nodevec_a2, self.nodevec_a3, self.nodevec_ak)
        adp_b = self.dgconstruct(self.nodevec_bp2[ind], self.nodevec_bp2, self.nodevec_bp3, self.nodevec_bpk)

        new_supports = [adp]
        new_supports_a = [adp_a]
        new_supports_b = [adp_b]

        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            # tcn for primary part
            residual = x
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate # [64, 32, 491, 12] t

            # tcn for a part
            residual_a = x_a

            filter_a = self.filter_convs_a[i](residual_a)
            # filter_a = self.filter_convs[i](residual_a)
            filter_a = torch.tanh(filter_a)
            gate_a = self.gate_convs_a[i](residual_a)
            # gate_a = self.gate_convs[i](residual_a)
            gate_a = torch.sigmoid(gate_a)
            x_a = filter_a * gate_a

            # tcn for b part
            residual_b = x_b
            # filter_b = self.filter_convs_b[i](residual_b)
            filter_b = self.filter_convs[i](residual_b)
            filter_b = torch.tanh(filter_b)
            # gate_b = self.gate_convs_b[i](residual_b)
            gate_b = self.gate_convs[i](residual_b)
            gate_b = torch.sigmoid(gate_b)
            x_b = filter_b * gate_b


            # skip connection * 3
            s = x
            s = self.skip_convs[i](s)
            s_a = x_a
            s_a = self.skip_convs_a[i](s_a)
            s_b = x_b
            s_b = self.skip_convs_b[i](s_b)

            if isinstance(skip, int):  # B F N T
                skip = s.transpose(2, 3).reshape([s.shape[0], -1, s.shape[2], 1]).contiguous()
            else:
                skip = torch.cat([s.transpose(2, 3).reshape([s.shape[0], -1, s.shape[2], 1]), skip], dim=1).contiguous()

            if isinstance(skip_a, int):  # B F N T
                skip_a = s_a.transpose(2, 3).reshape([s_a.shape[0], -1, s_a.shape[2], 1]).contiguous()
            else:
                skip_a = torch.cat([s_a.transpose(2, 3).reshape([s_a.shape[0], -1, s_a.shape[2], 1]), skip_a], dim=1).contiguous()

            if isinstance(skip_b, int):  # B F N T
                skip_b = s_b.transpose(2, 3).reshape([s_b.shape[0], -1, s_b.shape[2], 1]).contiguous()
            else:
                skip_b = torch.cat([s_b.transpose(2, 3).reshape([s_b.shape[0], -1, s_b.shape[2], 1]), skip_b], dim=1).contiguous()


            residual_graph = x
            residual_graph_a = x_a
            residual_graph_b = x_b
            x = self.normal_graph[i](x)
            x_a = self.normal_graph_a[i](x_a)
            x_b = self.normal_graph_b[i](x_b)
            x = self.gconv[i](x, new_supports)  # [64,32,491,12]
            x_a = self.gconv_a[i](x_a, new_supports_a)
            x_b = self.gconv_b[i](x_b, new_supports_b)
            x = x + residual_graph[:, :, :, -x.size(3):]
            x_a = x_a + residual_graph_a[:, :, :, -x_a.size(3):]
            x_b = x_b + residual_graph_b[:, :, :, -x_b.size(3):]

            x,_,_ = self.attenLayer[i](x, x_a, x_b)
            x_a,_,_ = self.attenLayer_a[i](x_a, x, x_b)
            x_b,_,_ = self.attenLayer_b[i](x_b, x, x_a)

            x = x + residual[:, :, :, -x.size(3):]
            x_a = x_a + residual_a[:, :, :, -x_a.size(3):]
            x_b = x_b + residual_b[:, :, :, -x_b.size(3):]

            x = self.normal[i](x)
            x_a = self.normal_a[i](x_a)
            x_b = self.normal_b[i](x_b)
        # output layer
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        x_a = F.relu(skip_a)
        x_a = F.relu(self.end_conv_1_a(x_a))
        x_a = self.end_conv_2_a(x_a)

        x_b = F.relu(skip_b)
        x_b = F.relu(self.end_conv_1_b(x_b))
        x_b = self.end_conv_2_b(x_b)
        prediction = torch.cat([x, x_a, x_b], dim=-1)
        result={}
        result['prediction'] = prediction
        result['log_sigma'] = self.log_sigma
        # DMA
        # result['weights'] = self.weights
        return result
 