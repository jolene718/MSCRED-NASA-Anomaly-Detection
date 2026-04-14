from __future__ import annotations

import warnings

from model.mscred_nasa import MSCRED as _NASAMSCRED


class MSCRED(_NASAMSCRED):
    """
    Backward-compatible wrapper around the NASA-adapted MSCRED implementation.

    Older project code instantiated `MSCRED(in_channels_encoder=3, in_channels_decoder=256)`.
    The adapted model only needs the number of signature-matrix scales, so we accept
    both signatures here and route everything to the variable-size implementation.
    """

    def __init__(
        self,
        input_channels: int | None = None,
        in_channels_encoder: int | None = None,
        in_channels_decoder: int | None = None,
    ) -> None:
        resolved_channels = input_channels if input_channels is not None else in_channels_encoder
        if resolved_channels is None:
            raise ValueError("MSCRED requires `input_channels` or `in_channels_encoder`.")

        if in_channels_decoder is not None and in_channels_decoder != 256:
            warnings.warn(
                "The NASA-adapted decoder infers its hidden width automatically; "
                "`in_channels_decoder` is kept only for backward compatibility.",
                stacklevel=2,
            )

        super().__init__(input_channels=int(resolved_channels))


__all__ = ["MSCRED"]

"""
import torch
import torch.nn as nn #pyTorch用于构建神经网络nn
import numpy as np
from model.convolution_lstm import ConvLSTM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #检测有无GPU。

def attention(ConvLstm_out): #用attention融合5个时间步
    attention_w = [] #因为融合需要关注权重 ——> 记录每个样本中，最后一个时间步和其他时间步 之间的关系 ——> 融合5个时间步
    for k in range(5): #每个样本有5个时间步
        attention_w.append(torch.sum(torch.mul(ConvLstm_out[k], ConvLstm_out[-1]))/5) #数值越大代表两个时间步关系越深；ConvLstm_out[k]:第k个时间步的特征图；k=-1:最后一个时间步的特征图；torch.mul()两个时间步的特征图对应位置相加
    m = nn.Softmax() #m即Softmax，将5个数字（张量形态）转成权重比例
    attention_w = torch.reshape(m(torch.stack(attention_w)), (-1, 5)) #torch.stack() 将5个零散数字转成张量tensor；reshape..(-1,5)是为了把关系值张量形状定为?行5列
    cl_out_shape = ConvLstm_out.shape #因为之后要把数据还原成原来的形状所以存个档 (5,32,30,30) 5:时间步 32:过滤器数？ 30:传感器数
    ConvLstm_out = torch.reshape(ConvLstm_out, (5, -1)) #改为5行?列，也就是除了保留时间步维度，剩余拉成一维
    convLstmOut = torch.matmul(attention_w, ConvLstm_out) #(?,5)和(5,?)矩阵乘法，乘完5和5对消变1——>恢复形状时不需要时间步维度（cl_out_shape[0]）了。and 这时候就融合了5个时间步。
    convLstmOut = torch.reshape(convLstmOut, (cl_out_shape[1], cl_out_shape[2], cl_out_shape[3])) #还原其他维度。
    return convLstmOut

class CnnEncoder(nn.Module): #大矩阵——>[CNN编码器压缩信息]——>小小的特征图。CLASS定义一个“模具”
    def __init__(self, in_channels_encoder): #__xx_：系统自带的构造函数，创建CLASS时第一时间自动运行的初始化函数；in_channels_encoder：输入数据的通道数
        super(CnnEncoder, self).__init__() #super：调用父类的构造函数，即加载nn.Module的功能
        self.conv1 = nn.Sequential( #nn.Sequential把多层按序打包
            nn.Conv2d(in_channels_encoder, 32, 3, (1, 1), 1), #Conv2d：2维卷积层。(输入通道数——>3因为有3种窗口大小,输出通道数32,卷积核大小——>3*3,步长——>每次移动1格,填充——>周围补一圈0)
            nn.SELU() #激活函数
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, (2, 2), 1),
            nn.SELU()
        )    
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 2, (2, 2), 1),
            nn.SELU()
        )   
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 2, (2, 2), 0), #...每次移动2格，周围不补0
            nn.SELU()
        )
    def forward(self, X): #pyTorch要求必须明确规定数据前向传播的路径
        conv1_out = self.conv1(X)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        return conv1_out, conv2_out, conv3_out, conv4_out #返回4层不同大小的特征图


class Conv_LSTM(nn.Module):
    def __init__(self):
        super(Conv_LSTM, self).__init__()
        self.conv1_lstm = ConvLSTM(input_channels=32, hidden_channels=[32],  #输入通道数，记忆通道数(让LSTM记住32层特征)
                                   kernel_size=3, step=5, effective_step=[4]) #卷积核大小，时间步长度，只输出第5个时间步的结果
        self.conv2_lstm = ConvLSTM(input_channels=64, hidden_channels=[64], 
                                   kernel_size=3, step=5, effective_step=[4])
        self.conv3_lstm = ConvLSTM(input_channels=128, hidden_channels=[128], 
                                   kernel_size=3, step=5, effective_step=[4])
        self.conv4_lstm = ConvLSTM(input_channels=256, hidden_channels=[256], 
                                   kernel_size=3, step=5, effective_step=[4])

    def forward(self, conv1_out, conv2_out, 
                conv3_out, conv4_out):
        conv1_lstm_out = self.conv1_lstm(conv1_out)
        conv1_lstm_out = attention(conv1_lstm_out[0][0]) #[0][0]拿到最里面真正的 5 个时间步特征图
        conv2_lstm_out = self.conv2_lstm(conv2_out)
        conv2_lstm_out = attention(conv2_lstm_out[0][0])
        conv3_lstm_out = self.conv3_lstm(conv3_out)
        conv3_lstm_out = attention(conv3_lstm_out[0][0])
        conv4_lstm_out = self.conv4_lstm(conv4_out)
        conv4_lstm_out = attention(conv4_lstm_out[0][0])
        return conv1_lstm_out.unsqueeze(0), conv2_lstm_out.unsqueeze(0), conv3_lstm_out.unsqueeze(0), conv4_lstm_out.unsqueeze(0) #.unsqueeze(0):在最前面加一个维度

class CnnDecoder(nn.Module):
    def __init__(self, in_channels):
        super(CnnDecoder, self).__init__()
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, 2, 2, 0, 0), #反卷积(输入通道数——>256,输出通道数,核大小——>2*2,步长——>2,padding和output_padding——>0代表单纯放大）
            nn.SELU()
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 2, 2, 1, 1),
            nn.SELU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 3, 2, 1, 1), #...padding=1四周都补0,output_padding=1只补 右边 + 下边 各 1 行 / 列
            nn.SELU()
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 3, 1, 1, 0),
            nn.SELU()
        )
    
    def forward(self, conv1_lstm_out, conv2_lstm_out, conv3_lstm_out, conv4_lstm_out):
        deconv4 = self.deconv4(conv4_lstm_out)
        deconv4_concat = torch.cat((deconv4, conv3_lstm_out), dim = 1) #dim=1 = 在通道维度拼接——>所谓SKIP CONNECTION——>把解码器当前结果 + 编码器对应层的特征 拼在一起！
        deconv3 = self.deconv3(deconv4_concat)
        deconv3_concat = torch.cat((deconv3, conv2_lstm_out), dim = 1) #因为编码器存了原始细节，解码器只存了抽象特征，拼起来才能精准还原矩阵，不然模糊不清
        deconv2 = self.deconv2(deconv3_concat)
        deconv2_concat = torch.cat((deconv2, conv1_lstm_out), dim = 1)
        deconv1 = self.deconv1(deconv2_concat)
        return deconv1


class MSCRED(nn.Module): #总的MSCRED模型 CLASS
    def __init__(self, in_channels_encoder, in_channels_decoder): #in_channels_encoder：编码器输入通道数 = 3（你 3 个尺度）；in_channels_decoder：解码器输入通道数 = 256（编码器最后一层输出）
        super(MSCRED, self).__init__() #启用 pytorch 模型功能（训练、测试、保存、加载…）
        self.cnn_encoder = CnnEncoder(in_channels_encoder) #把三个小零件，装到总模型里！
        self.conv_lstm = Conv_LSTM()
        self.cnn_decoder = CnnDecoder(in_channels_decoder)
    
    def forward(self, x): #输入 x 形状：(5, 3, 30, 30) —— 5个时间步，3个尺度，30×30矩阵
        conv1_out, conv2_out, conv3_out, conv4_out = self.cnn_encoder(x) #输出4层特征，越来越小
        conv1_lstm_out, conv2_lstm_out, conv3_lstm_out, conv4_lstm_out = self.conv_lstm( #4层特征分别过 ConvLSTM；用 Attention 选重要的时间步；输出带时序信息的特征
                                conv1_out, conv2_out, conv3_out, conv4_out) 

        gen_x = self.cnn_decoder(conv1_lstm_out, conv2_lstm_out, 
                                conv3_lstm_out, conv4_lstm_out) #从深层到浅层，逐步还原；用 Skip Connection 补充细节；输出 (1, 3, 30, 30) —— 重建的矩阵！
        return gen_x #gen_x = 重建的最后一步矩阵


#encoder输入：每个样本 15个30*30大小的相似度矩阵； decorder输出：神经网络学到（卷积提取）的 “特征图”
"""
