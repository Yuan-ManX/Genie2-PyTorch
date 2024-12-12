from __future__ import annotations

from math import ceil, sqrt
from random import random
from functools import partial

import torch
from torch import nn, tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torchvision.utils import save_image

import einx
from einops import rearrange, reduce, repeat, pack, unpack

from vector_quantize_pytorch import VectorQuantize, ResidualVQ
from x_transformers.x_transformers import RotaryEmbedding
from x_transformers import Decoder, AutoregressiveWrapper
from imagen_pytorch import Imagen


# tensor typing
# 张量类型

import jaxtyping
from beartype import beartype

class TorchTyping:
    """
    TorchTyping 类用于简化 Tensor 数据类型的定义，支持指定形状。

    初始化参数:
        abstract_dtype (Type): 抽象数据类型，例如 Float, Int, Bool。
    """
    def __init__(self, abstract_dtype):
        # 存储抽象数据类型
        self.abstract_dtype = abstract_dtype

    def __getitem__(self, shapes: str):
        """
        根据指定的形状字符串返回带有形状信息的 Tensor 数据类型。

        参数:
            shapes (str): 形状字符串，例如 "batch, channels, height, width"。

        返回:
            Type: 带有形状信息的 Tensor 数据类型，例如 Float[Tensor, "batch, channels, height, width"]。
        """
        # 返回带有形状信息的 Tensor 数据类型
        return self.abstract_dtype[Tensor, shapes]

# 定义 Float, Int, Bool 数据类型，支持指定形状
Float = TorchTyping(jaxtyping.Float)
Int   = TorchTyping(jaxtyping.Int)
Bool  = TorchTyping(jaxtyping.Bool)


# einstein notation
# einstein符号

# b - batch
# c - channels
# t - time
# h - height
# w - width
# n - sequence (flattened latent time * height * width)
# s - space sequence
# l - logits
# a - number of actions (multiple keys pressed)


# helper functions
# 辅助函数

def exists(v):
    """
    检查一个值是否存在（不为 None）。

    参数:
        v (Optional[Any]): 需要检查的值。

    返回:
        bool: 如果 v 不为 None，则返回 True；否则返回 False。
    """
    return v is not None


def default(v, d):
    """
    返回可选值或默认值。

    参数:
        v (Optional[Any]): 需要检查的可选值。
        d (Any): 默认值。

    返回:
        Any: 如果 v 存在，则返回 v；否则返回 d。
    """
    return v if exists(v) else d


def identity(t):
    """
    返回输入张量本身。

    参数:
        t (Tensor): 输入张量。

    返回:
        Tensor: 输入张量 t。
    """
    return t


def l2norm(t, dim = -1):
    """
    对张量 t 在指定的维度 dim 上进行 L2 归一化。

    参数:
        t (Tensor): 输入张量。
        dim (int, 可选): 需要归一化的维度，默认为 -1。

    返回:
        Tensor: 归一化后的张量。
    """
    return F.normalize(t, dim = dim, p = 2)


def lens_to_mask(lens, total_len):
    """
    根据序列长度生成掩码张量。

    参数:
        lens (Tensor): 每个序列的长度，形状为 (batch_size,)。
        total_len (int): 序列的最大长度。

    返回:
        Tensor: 掩码张量，形状为 (batch_size, total_len)。如果序列长度小于 total_len，则对应位置为 False。
    """
    # 生成一个从 0 到 total_len-1 的序列张量
    seq = torch.arange(total_len, device = lens.device)
    # 生成掩码，形状为 (batch_size, total_len)
    return einx.less('n, b -> b n', seq, lens)


def pack_one(t, pattern):
    """
    将单个张量 t 按照指定的 pattern 打包，并返回打包后的张量和打包参数。

    参数:
        t (Tensor): 需要打包的张量。
        pattern (str): 打包的模式，例如 'b *' 表示批次维度和其他维度。

    返回:
        Tuple[Tensor, Tuple]: 返回打包后的张量和打包参数。
    """
    # 按照指定的 pattern 打包张量 t
    packed, ps = pack([t], pattern)

    def inverse(out, inv_pattern = None):
        """
        反向操作，将打包后的张量 out 解包回原始张量。

        参数:
            out (Tensor): 打包后的张量。
            inv_pattern (Optional[str], 可选): 反向解包的 pattern。如果未提供，则使用原始的 pattern。

        返回:
            Tensor: 解包后的原始张量。
        """
        # 如果未提供 inv_pattern，则使用原始的 pattern
        inv_pattern = default(inv_pattern, pattern)
        # 解包并返回第一个张量
        return unpack(out, ps, inv_pattern)[0]
    # 返回打包后的张量和反向操作函数
    return packed, inverse


def project(x, y):
    """
    将张量 x 投影到张量 y 上，并返回平行分量和正交分量。

    参数:
        x (Tensor): 输入张量，形状为 (batch_size, ...)。
        y (Tensor): 投影目标张量，形状为 (batch_size, ...)。

    返回:
        Tuple[Tensor, Tensor]: 返回平行分量和正交分量的张量。
    """
    # 将张量 x 按照 'b *' 模式打包，并获取反向操作函数
    x, inverse = pack_one(x, 'b *')
    # 将张量 y 按照 'b *' 模式打包
    y, _ = pack_one(y, 'b *')

    # 获取张量 x 的数据类型
    dtype = x.dtype
    # 将张量 x 和 y 转换为双精度浮点数类型
    x, y = x.double(), y.double()
    # 对张量 y 在最后一个维度上进行 L2 归一化，得到单位向量
    unit = l2norm(y, dim = -1)

    # 计算平行分量：x 在 y 方向上的投影
    parallel = (x * unit).sum(dim = -1, keepdim = True) * unit
    # 计算正交分量：x 中垂直于 y 的部分
    orthogonal = x - parallel

    # 将平行分量和正交分量反向解包回原始数据类型
    return inverse(parallel).to(dtype), inverse(orthogonal).to(dtype)


# input action related helprs
# 输入操作相关的内容

def valid_action_input(inp):
    """
    验证输入字符串是否为有效的动作输入。

    参数:
        inp (str): 输入字符串，格式应为逗号分隔的数字，例如 "1,2,3"。

    返回:
        bool: 如果输入字符串的每个部分都是数字，则返回 True；否则返回 False。
    """
    # 将输入字符串按逗号分割成列表
    inp = inp.split(',')
    # 检查每个元素是否为数字
    return all(i.strip().isdigit() for i in inp)


# sampling helpers
# 采样

def log(t, eps = 1e-20):
    """
    计算张量 t 的对数，并避免数值下溢。

    参数:
        t (Tensor): 输入张量。
        eps (float, 可选): 最小值阈值，用于避免数值下溢。默认为 1e-20。

    返回:
        Tensor: 对数后的张量。
    """
    # 计算对数，并对张量进行最小值裁剪
    return torch.log(t.clamp(min = eps))


def gumbel_noise(t):
    """
    生成与张量 t 形状相同的 Gumbel 噪声。

    参数:
        t (Tensor): 输入张量，用于确定噪声的形状。

    返回:
        Tensor: 生成的 Gumbel 噪声张量。
    """
    # 生成与 t 形状相同的均匀噪声，范围 [0, 1)
    noise = torch.zeros_like(t).uniform_(0, 1)
    # 应用 Gumbel 变换，生成 Gumbel 噪声
    return -log(-log(noise))


def gumbel_sample(t, temperature = 1., dim = -1, keepdim = True):
    """
    使用 Gumbel-Softmax 采样方法对张量 t 进行采样。

    参数:
        t (Tensor): 输入张量，通常是 logits。
        temperature (float, 可选): 温度参数，控制采样的软化程度。默认为 1.0。
        dim (int, 可选): 进行 argmax 操作的维度。默认为最后一个维度。
        keepdim (bool, 可选): 是否保留维度。默认为 True。

    返回:
        Tensor: 采样后的张量。
    """
    # 应用 Gumbel-Softmax 采样
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim, keepdim = keepdim)


# min_p
# https://arxiv.org/abs/2407.01082

def min_p_filter(logits, min_p = 0.1):
    """
    对 logits 应用最小概率过滤。

    参数:
        logits (Tensor): 输入的 logits 张量，形状为 (batch_size, ..., num_classes)。
        min_p (float, 可选): 最小概率阈值。默认值为 0.1。

    返回:
        Tensor: 应用过滤后的 logits 张量。如果某个类别的概率小于 min_p * 最大概率，则将其对应的 logit 设置为负无穷大。
    """
    # 计算每个样本在所有类别上的 softmax 概率
    probs = logits.softmax(dim = -1) # 计算 softmax 概率，形状为 (batch_size, ..., num_classes)

    # 找到每个样本的最大概率，并保持维度以便后续广播
    max_probs = probs.amax(dim = -1, keepdim = True) # 获取每个样本的最大概率，形状为 (batch_size, ..., 1)

    # 计算概率阈值：min_p 乘以每个样本的最大概率
    limit = min_p * max_probs # 计算每个样本的概率阈值，形状为 (batch_size, ..., 1)

    # 对 logits 应用过滤：如果某个类别的概率小于阈值，则将其对应的 logit 设置为负无穷大
    return torch.where(probs < limit, float('-inf'), logits) # 返回过滤后的 logits


# wrapper for adding meta tokens

class MetaTokenWrapper(Module):
    """
    MetaTokenWrapper 类用于在输入张量前添加元标记。

    初始化参数:
        fn (Decoder): 解码器模型，用于处理添加元标记后的输入。
        num_meta_tokens (int): 元标记的数量。
    """
    def __init__(
        self,
        fn: Decoder,
        num_meta_tokens
    ):
        super().__init__()
        # 存储解码器模型
        self.fn = fn
        # 定义元标记参数，形状为 (num_meta_tokens, dim)
        self.meta_tokens = nn.Parameter(torch.zeros(num_meta_tokens, fn.dim))

    def forward(self, x, *args, **kwargs):
        """
        前向传播方法，在输入张量前添加元标记。

        参数:
            x (Tensor): 输入张量，形状为 (batch_size, ..., dim)。
            *args: 其他位置参数，传递给解码器模型。
            **kwargs: 其他关键字参数，传递给解码器模型。

        返回:
            Tensor: 解码器模型的输出张量。
        """
        # 重复元标记张量，使其与输入张量的批次大小相同
        meta_tokens = repeat(self.meta_tokens, '... -> b ...', b = x.shape[0]) # 形状变为 (batch_size, num_meta_tokens, dim)

        # 将元标记和输入张量打包在一起，形状变为 (batch_size, num_meta_tokens + input_dim)
        x, packed_shape = pack([meta_tokens, x], 'b * d')

        # 将打包后的张量传递给解码器模型进行处理
        out = self.fn(x, *args, **kwargs)

        # 将解码器模型的输出张量解包，恢复原始输入张量的形状
        _, out = unpack(out, packed_shape, 'b * d')

        # 返回解码器模型的输出张量
        return out


# main class

class Genie2(Module):
    """
    Genie2 类是一个生成模型，融合了编码器、解码器、向量量化（VQ）和动作嵌入。

    参数:
        dim (int): 模型的基本维度。
        dim_latent (int): 潜在空间的维度。
        num_actions (int | None, 可选): 动作的数量。如果为 None，则不使用动作嵌入。
        depth (int, 可选): Transformer 的深度（层数），默认为 12。
        attn_dim_head (int, 可选): 注意力头的维度，默认为 64。
        heads (int, 可选): 注意力头的数量，默认为 8。
        latent_channel_first (bool, 可选): 潜在张量是否以通道优先的形式存储，默认为 False。
        cfg_train_action_dropout (float, 可选): 在训练时动作条件的 dropout 概率，默认为 0.5。
        transformer_kwargs (dict, 可选): 传递给 Transformer 的关键字参数，默认为：
            - add_value_residual (bool): 是否添加值残差，默认为 True。
            - learned_value_residual_mix (bool): 是否使用学习到的值残差混合，默认为 True。
            - ff_glu (bool): 是否使用 GLU 激活函数，默认为 True。
            - use_rmsnorm (bool): 是否使用 RMSNorm 归一化，默认为 True。
        action_transformer_kwargs (dict, 可选): 传递给动作 Transformer 的关键字参数，默认为：
            - add_value_residual (bool): 是否添加值残差，默认为 True。
            - learned_value_residual_mix (bool): 是否使用学习到的值残差混合，默认为 True。
            - ff_glu (bool): 是否使用 GLU 激活函数，默认为 True。
            - use_rmsnorm (bool): 是否使用 RMSNorm 归一化，默认为 True。
            - depth (int): Transformer 的深度，默认为 2。
            - heads (int): 注意力头的数量，默认为 4。
            - attn_dim_head (int): 注意力头的维度，默认为 64。
        num_meta_tokens (int, 可选): 元标记的数量，默认为 16。用于 Hymba 模型。
        vq_codebook_size (int, 可选): 向量量化器的码本大小，默认为 4096。
        vq_kwargs (dict, 可选): 传递给向量量化器的关键字参数。
        encoder (nn.Module, 可选): 编码器模型，默认为恒等映射。
        decoder (nn.Module, 可选): 解码器模型，默认为恒等映射。
        vq_commit_loss_weight (float, 可选): 向量量化器的承诺损失权重，默认为 1.0。
        allow_multiple_actions (bool, 可选): 是否允许多个动作，默认为 False。
        max_num_actions (int, 可选): 最大动作数量，默认为 10。
        action_autoregressive_loss_weight (float, 可选): 动作自回归损失的权重，默认为 0.1。
        is_video_enc_dec (bool, 可选): 是否使用视频编码器和解码器，默认为 False。默认为图像编码器/解码器，但未来可能会使用视频扩散模型。
    """
    @beartype
    def __init__(
        self,
        dim, # 模型的基本维度
        dim_latent, # 潜在空间的维度
        num_actions: int | None = None, # 动作的数量，如果为 None，则不使用动作嵌入
        depth = 12, # Transformer 的深度（层数）
        attn_dim_head = 64, # 注意力头的维度
        heads = 8, # 注意力头的数量
        latent_channel_first = False, # 潜在张量是否以通道优先的形式存储
        cfg_train_action_dropout = 0.5, # 在训练时动作条件的 dropout 概率
        transformer_kwargs: dict = dict( # Transformer 的关键字参数
            add_value_residual = True,
            learned_value_residual_mix = True,
            ff_glu = True,
            use_rmsnorm = True,
        ),
        action_transformer_kwargs: dict = dict( # 动作 Transformer 的关键字参数
            add_value_residual = True,
            learned_value_residual_mix = True,
            ff_glu = True,
            use_rmsnorm = True,
            depth = 2,
            heads = 4,
            attn_dim_head = 64
        ),
        # 元标记的数量，用于 Hymba 模型
        num_meta_tokens = 16, # meta tokens used in Hymba https://www.arxiv.org/abs/2411.13676
        vq_codebook_size = 4096, # 向量量化器的码本大小
        vq_kwargs: dict = dict(), # 向量量化器的关键字参数
        encoder: Module = nn.Identity(), # 编码器模型，默认为恒等映射
        decoder: Module = nn.Identity(), # 解码器模型，默认为恒等映射
        vq_commit_loss_weight = 1., # 向量量化器的承诺损失权重
        allow_multiple_actions = False, # 是否允许多个动作
        max_num_actions = 10, # 最大动作数量
        action_autoregressive_loss_weight = 0.1, # 动作自回归损失的权重
        # 是否使用视频编码器和解码器，默认为 False
        is_video_enc_dec = False # by default will assume image encoder / decoder, but in the future, video diffusion models with temporal compression will likely perform even better, imo
    ):
        super().__init__()

        # 存储动作数量
        self.num_actions = num_actions
        # 定义动作嵌入层，如果 num_actions 为 None，则不使用动作嵌入
        self.action_embed = nn.Embedding(num_actions, dim) if exists(num_actions) else None

        # 存储编码器模型
        self.encoder = encoder
        # 存储解码器模型
        self.decoder = decoder

        # 存储是否使用视频编码器和解码器
        self.is_video_enc_dec = is_video_enc_dec

        # 存储潜在空间的维度
        self.dim_latent = dim_latent
        # 存储潜在张量是否以通道优先的形式存储
        self.latent_channel_first = latent_channel_first

        # 定义从潜在空间到模型维度的线性层
        self.latent_to_model = nn.Linear(dim_latent, dim)
        # 定义从模型维度到潜在空间的线性层
        self.model_to_latent = nn.Linear(dim, dim_latent)

        # 定义时间旋转嵌入
        self.time_rotary = RotaryEmbedding(
            dim = attn_dim_head // 2 # 旋转嵌入的维度
        )

        # 定义向量量化器
        self.vq = VectorQuantize(
            dim = dim_latent, # 潜在空间的维度
            codebook_size = vq_codebook_size, # 码本大小
            rotation_trick = False, # 是否使用旋转技巧
            **vq_kwargs # 其他关键字参数
        )

        # 存储向量量化器的承诺损失权重
        self.vq_commit_loss_weight = vq_commit_loss_weight

        # wrapper for adding meta tokens

        # 存储元标记的数量
        self.num_meta_tokens = num_meta_tokens
        # 如果元标记数量大于 0，则使用 MetaTokenWrapper 进行包装；否则，使用恒等映射
        meta_token_wrapper = partial(MetaTokenWrapper, num_meta_tokens = num_meta_tokens) if num_meta_tokens > 0. else identity

        # main "world model" dynamics model transformer
        # 世界模型 Transformer

        # 使用元标记包装器包装 Transformer
        self.transformer = meta_token_wrapper(Decoder( # 定义解码器模型
            dim = dim, # 模型的基本维度
            depth = depth, # Transformer 的深度（层数）
            heads = heads, # 注意力头的数量
            attn_dim_head = attn_dim_head, # 注意力头的维度
            **transformer_kwargs # 其他 Transformer 参数
        ))

        # action related
        # 与动作相关的部分

        # 是否允许多个动作
        self.allow_multiple_actions = allow_multiple_actions
        # 如果允许多个动作，则定义最大动作数量
        self.max_num_actions = max_num_actions # in the case multiple actions are allowed, maximum number of actions allowed

        # 判断是否需要动作自回归损失
        has_action_loss = action_autoregressive_loss_weight > 0.
        # 存储是否需要动作自回归损失
        self.has_action_loss = has_action_loss

        # 初始化动作预测层
        self.to_action_pred = None

        if has_action_loss:
            if allow_multiple_actions:
                # 定义动作 Transformer 的维度
                dim_action_transformer = dim // 2

                # 定义动作结束符的 ID
                self.action_eos_id = num_actions
                # 定义动作位置嵌入参数
                self.action_pos_embed = nn.Parameter(torch.zeros(max_num_actions, dim))

                # 定义动作预测层
                self.to_action_pred = nn.Sequential(
                    # 线性层，将模型维度映射到动作 Transformer 的维度
                    nn.Linear(dim, dim_action_transformer, bias = False),
                    # 使用元标记包装器包装动作 Transformer
                    meta_token_wrapper(dim, Decoder(
                        dim = dim_action_transformer, # 动作 Transformer 的维度
                        **action_transformer_kwargs # 其他动作 Transformer 参数
                    )),
                    # 线性层，将动作 Transformer 的输出映射到动作数量 + 1（结束符）
                    nn.Linear(dim_action_transformer, num_actions + 1, bias = False)
                )
            else:
                # 如果不允许多个动作，则定义线性层，将模型维度映射到动作数量
                self.to_action_pred = nn.Linear(dim, num_actions, bias = False)

        # 存储动作自回归损失的权重
        self.action_autoregressive_loss_weight = action_autoregressive_loss_weight

        # 注册一个缓冲区，存储零张量，用于占位
        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # needed for classifier free guidance
        # 分类器引导（CFG）

        # 存储训练时动作条件的 dropout 概率，用于分类器引导（CFG）
        self.cfg_train_action_dropout = cfg_train_action_dropout

    # 梯度计算，提高推理速度并节省内存
    @torch.no_grad()
    def generate(
        self,
        image: Float['b c h w'], # 输入图像张量，形状为 (batch_size, channels, height, width)
        num_frames: int, # 要生成的视频帧数
        filter_kwargs: dict = dict(), # 传递给滤波器的关键字参数
        temperature = 0.9, # 生成动作时的温度参数，控制采样的随机性
        init_action: int | None = None, # 初始动作，如果为 None，则随机初始化
        interactive = False, # 是否为交互式生成模式
        interactive_save_last_frame = True, # 在交互模式下是否保存最后一帧图像
        cond_scale = 1., # 条件缩放因子，控制条件信息的强度
        **model_forward_kwargs # 其他传递给模型前向传播的关键字参数
    ):
        # 保存模型当前的训练状态
        was_training = self.training
        # 将模型设置为评估模式，禁用 dropout 和 batch normalization 的统计更新
        self.eval()

        # if interactive is set to True, only allow for sampling one video trajectory at a time
        # 如果是交互式生成模式，则只允许一次生成一个视频轨迹

        if interactive:
            # 确保输入图像的批次大小为 1
            assert image.shape[0] == 1
            # 确保提供了初始动作
            assert exists(init_action), f'init_action must be given as an integer from 0 - {self.num_actions - 1}'

            # 将初始动作转换为张量，形状为 (1, 1, 1)
            actions = tensor([[[init_action]]], device = self.device)
            # 设置最大动作数量为 1
            max_actions = 1

        else:
            # 如果不是交互式生成模式，则不初始化动作
            actions = None

        # ready image as single frame video
        # 将单帧图像准备为单帧视频

        # 重塑图像张量形状为 (batch_size, channels, 1, height, width)，表示单帧视频
        single_frame = rearrange(image, 'b c h w -> b c 1 h w')

        # encode single frame
        # 编码单帧图像

        # 编码单帧图像，获取编码后的状态和第一帧代码
        _, first_frame_code, _ = self.encode_state(single_frame)

        # store all latent codes
        # 存储所有潜在

        # 获取空间序列长度
        space_seq_len = first_frame_code.shape[-1]
        # 计算高度，假设空间序列长度为平方数
        height = int(sqrt(space_seq_len)) # assume square for now

        # 初始化状态代码为第一帧
        state_codes = first_frame_code

        # autoregressive sample for number of frames
        # 自回归采样生成指定数量的帧

        for frame in range(1, num_frames + 1):

            if interactive:

                # before prompting the human model, show the human the last image from the world model
                # 在提示人类模型之前，显示世界模型生成的最后一帧图像

                if interactive_save_last_frame:
                    # 重塑状态形状为 (batch_size, time_steps, height, width)
                    unpacked_codes = rearrange(state_codes, 'b (t h w) -> b t h w', h = height, w = height)
                    # 从最后一帧的索引中获取潜在
                    last_frame_tokens = self.vq.get_codes_from_indices(unpacked_codes[:, -1])

                    if self.latent_channel_first:
                        # 如果潜在通道优先，则重塑潜在形状
                        last_frame_tokens = rearrange(last_frame_tokens, 'b ... d -> b d ...')

                    # 解码最后一帧潜在
                    last_frame = self.decoder(last_frame_tokens)
                    # 获取最后一帧图像并移动到 CPU
                    last_frame = last_frame[0].cpu().detach()
                    # 获取图像通道数
                    channels = last_frame.shape[0]

                    # 如果通道数小于等于 4，则认为是有效的图像类型
                    if channels <= 4: # assume valid image type if feature dimension is 4 or less
                        # 将图像像素值限制在 [0, 1] 范围内
                        last_frame.clamp_(0., 1.)
                        # 保存图像为 PNG 文件
                        save_image(last_frame, './last-frame.png')
                    else:
                        # 否则，保存图像为 PyTorch 文件
                        torch.save(last_frame, './last-frame.pt')

                # prompt human
                # 提示用户输入下一个动作
                
                while (maybe_next_action := input(f'[frame {frame}] enter the next action (0 - {self.num_actions}): ')) and not valid_action_input(maybe_next_action):
                    # 提示用户输入无效
                    print('invalid input, must be integer action - multiple actions need to be all integers separated by commas [ex. "1,3,24"]')

                # 将用户输入的字符串转换为整数列表
                maybe_next_actions = [*map(int, maybe_next_action.split(','))]
                # 去除重复的动作
                maybe_next_actions = [*set(maybe_next_actions)]

                if not self.allow_multiple_actions:
                    # 如果不允许多个动作，则确保只有一个动作
                    assert len(maybe_next_actions) == 1, f'you cannot interact with multiple actions if `allow_multiple_actions` is not set to `True`'
                else:
                    # 如果允许多个动作，则确保不超过最大动作数量
                    assert len(maybe_next_actions) <= self.max_num_actions, f'maximum number of actions is set at {self.max_num_actions}'

                # 将动作转换为张量
                next_action = tensor(maybe_next_actions, device = self.device)
                # 重塑张量形状为 (1, 1, a)
                next_action = rearrange(next_action, 'a -> 1 1 a')

                # 获取输入动作的数量
                input_num_actions = next_action.shape[-1]

                if input_num_actions > max_actions:
                    # 如果输入动作数量超过最大动作数量，则填充动作
                    actions = F.pad(actions, (0,  input_num_actions - max_actions), value = -1)
                    max_actions = input_num_actions
                elif input_num_actions < max_actions:
                    # 如果输入动作数量少于最大动作数量，则填充动作
                    next_action = F.pad(next_action, (0,  max_actions - input_num_actions), value = -1)

                # 将下一个动作拼接到动作序列中
                actions = torch.cat((actions, next_action), dim = 1)

            for _ in range(space_seq_len):
                # 使用分类器引导进行前向传播
                logits = self.forward_with_cfg(
                    state_codes = state_codes, # 当前状态代码
                    time_seq_len = frame + 1, # 当前时间步
                    actions = actions, # 当前动作序列
                    cond_scale = cond_scale, # 条件缩放因子
                    **model_forward_kwargs # 其他模型前向传播参数
                )

                # 获取最后一个 logit
                last_logit = logits[:, -1] 
                # 应用最小概率过滤
                last_logit = min_p_filter(last_logit, **filter_kwargs)
                # 使用 Gumbel-Softmax 采样方法采样
                sampled = gumbel_sample(last_logit, temperature = temperature)
                # 将采样结果打包到状态代码中
                state_codes, _ = pack([state_codes, sampled], 'b *')

        # get all the latent codes
        # 获取所有潜在tokens

        # 从状态tokens中获取潜在tokens
        tokens = self.vq.get_codes_from_indices(state_codes)

        # restore time and space dims
        # 恢复时间和空间维度

        # 重塑张量形状为 (batch_size, time_steps, height, width, channels)，其中 time_steps = num_frames + 1
        tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', t = num_frames + 1, h = height)

        if self.latent_channel_first:
            # 如果潜在通道优先，则重塑张量形状为 (batch_size, channels, ...)
            tokens = rearrange(tokens, 'b ... d -> b d ...')
        
        # 判断是否需要将时间维度折叠到批次维度中
        need_fold_time_into_batch = not self.is_video_enc_dec

        if need_fold_time_into_batch:
            # 如果不是视频编码器/解码器，则将时间维度折叠到批次维度中，重塑为 (batch_size, time_steps, channels, height, width)
            tokens = rearrange(tokens, 'b c t h w -> b t c h w')
            # 将时间维度打包到批次维度中
            tokens, unpack_time = pack_one(tokens, '* c h w')

        # decode back to video
        # 解码回视频

        # 使用解码器将潜在tokens解码回视频
        video = self.decoder(tokens)

        if need_fold_time_into_batch:
            # 如果需要，将时间维度从批次维度中解包
            video = unpack_time(video, '* c h w')
            # 重塑张量形状为 (batch_size, channels, time_steps, height, width)
            video = rearrange(video, 'b t c h w -> b c t h w')

        # 将模型恢复到之前的训练状态
        self.train(was_training)

        if not interactive:
            # 如果不是交互式生成模式，则返回生成的视频
            return video
        
        # 如果是交互式生成模式，则返回生成的视频和动作序列
        return video, actions

    def encode_state(
        self,
        state: Float['b c t h w'] # 输入状态张量，形状为 (batch_size, channels, time_steps, height, width)
    ):

        # only need to fold time into batch if not a video enc/dec (classic image enc/dec of today)
        # 如果不是视频编码器/解码器（经典的图像编码器/解码器），则需要将时间维度折叠到批次维度中
        """
        将输入状态编码为潜在代码。

        参数:
            state (Float['b c t h w']): 输入状态张量，形状为 (batch_size, channels, time_steps, height, width)。

        返回:
            Tensor: 编码后的潜在代码，形状为 (batch_size, channels, time_steps, height, width)。
        """
        # 判断是否需要折叠时间维度
        need_fold_time_into_batch = not self.is_video_enc_dec

        if need_fold_time_into_batch:
            # 重塑张量形状为 (batch_size, time_steps, channels, height, width)
            state = rearrange(state, 'b c t ... -> b t c ...')
            # 将时间维度打包到批次维度中，重塑为 (batch_size * time_steps, channels, height, width)
            state, unpack_time = pack_one(state, '* c h w') # state packed into images

        # encode into latents
        # 将状态编码为潜在

        # 使用编码器将状态编码为潜在
        latents = self.encoder(state)

        if need_fold_time_into_batch:
            # 如果需要，将时间维度从批次维度中解包
            latents = unpack_time(latents, '* c h w')
            # 重塑张量形状为 (batch_size, channels, time_steps, height, width)
            latents = rearrange(latents, 'b t c h w -> b c t h w')

        # handle channel first, if encoder does not
        # 如果编码器没有这样做，处理通道优先

        if self.latent_channel_first:
            # 如果潜在通道优先，则重塑张量形状为 (batch_size, ..., channels)
            latents = rearrange(latents, 'b d ... -> b ... d')

        # pack time and spatial fmap into a sequence for transformer
        # 将时间和空间特征图打包成一个序列，用于 Transformer

        # 将时间和空间维度打包到批次维度中，重塑为 (batch_size, ..., channels)
        latents, unpack_time_space_dims = pack_one(latents, 'b * d')

        # 确保潜在代码的最后一个维度等于潜在空间的维度
        assert latents.shape[-1] == self.dim_latent

        # discrete quantize - offer continuous later, either using GIVT https://arxiv.org/abs/2312.02116v2 or Kaiming He's https://arxiv.org/abs/2406.11838
        # 离散量化 - 未来可以考虑使用 GIVT 或 Kaiming He's 的方法进行连续量化

        # 使用向量量化器对潜在代码进行量化，并返回量化后的潜在代码
        return self.vq(latents)

    @property
    def device(self):
        """
        获取模型当前使用的设备（CPU 或 GPU）。

        返回:
            torch.device: 模型参数的设备。
        """
        return next(self.parameters()).device

    def forward_with_cfg(
        self,
        *args,
        actions,
        cond_scale = 1.,
        parallel_keep_frac = 0.,
        **kwargs
    ):
        """
        使用条件缩放因子进行前向传播，并计算条件引导（CFG）输出。

        参数:
            *args: 其他位置参数，传递给前向传播方法。
            actions: 动作张量，用于条件生成。
            cond_scale (float, 可选): 条件缩放因子，控制条件信息的强度。默认为 1.0。
            parallel_keep_frac (float, 可选): 并行分量保留比例，用于调整并行分量。默认为 0.0。
            **kwargs: 其他关键字参数，传递给前向传播方法。

        返回:
            Tensor: 条件引导后的输出张量。
        """
        if not exists(actions):
            # 如果没有提供动作，则直接调用前向传播方法，不进行条件引导
            return self.forward(*args, return_loss = False, **kwargs)

        # 调用前向传播方法，传入动作，得到 logits
        logits = self.forward(
            *args,
            actions = actions,
            return_loss = False,
            **kwargs
        )

        if cond_scale == 1:
            # 如果条件缩放因子为 1，则直接返回 logits，不进行条件引导
            return logits

        # 调用前向传播方法，不传入动作，得到 null_logits（无条件的 logits）
        null_logits = self.forward(
            *args,
            actions = None,
            return_loss = False,
            **kwargs
        )

        # 计算 logits 和 null_logits 之间的差异
        update = logits - null_logits

        # 将差异向量分解为并行分量和正交分量
        parallel, orthog = project(update, logits)

        # 根据 parallel_keep_frac 保留并行分量，并清零正交分量
        update = parallel * parallel_keep_frac + orthog # 保留部分并行分量，并清零正交分量

        # 计算最终的 logits，添加条件引导后的更新
        return logits + update * (cond_scale - 1)

    def forward(
        self,
        state: Float['b c t h w'] | None = None, # 输入状态张量，形状为 (batch_size, channels, time_steps, height, width)
        state_codes: Int['b n'] = None, # 输入状态代码张量，形状为 (batch_size, n)
        time_seq_len: int | None = None, # 时间序列长度
        video_time_len: Int['b'] | None = None, # 视频时间长度，形状为 (batch_size,)
        actions: Int['b t'] | Int['b t a'] = None, # 动作张量，形状可以是 (batch_size, time_steps) 或 (batch_size, time_steps, num_actions)
        sort_actions = True, # 是否对动作进行排序，填充值为负数
        return_loss = True # 是否返回损失
    ):
        """
        前向传播方法，处理状态、状态代码、时间序列长度、视频时间长度和动作，生成潜在代码和动作嵌入。

        参数:
            state (Float['b c t h w'], 可选): 输入状态张量，形状为 (batch_size, channels, time_steps, height, width)。
            state_codes (Int['b n'], 可选): 输入状态代码张量，形状为 (batch_size, n)。
            time_seq_len (int, 可选): 时间序列长度。
            video_time_len (Int['b'], 可选): 视频时间长度，形状为 (batch_size,)。
            actions (Union[Int['b t'], Int['b t a']], 可选): 动作张量，形状可以是 (batch_size, time_steps) 或 (batch_size, time_steps, num_actions)。
            sort_actions (bool, 可选): 是否对动作进行排序，填充值为负数。默认为 True。
            return_loss (bool, 可选): 是否返回损失。默认为 True。

        返回:
            Tuple[Tensor, Tensor, Tensor, Tensor]: 返回量化后的潜在代码、潜在代码索引、承诺损失和动作嵌入。
        """
        # 确保提供了 state 或 state_codes 中的一个
        assert exists(state) ^ exists(state_codes)

        device = self.device

        if not exists(time_seq_len):
            assert exists(state)
            # 如果未提供 time_seq_len，则从 state 的形状中获取
            time_seq_len = state.shape[2]

        # 生成时间序列张量，范围从 0 到 time_seq_len-1
        time_seq = torch.arange(time_seq_len, device = device)

        # handle maybe variable lengthed videos
        # 处理可能长度可变的视频

        # 判断视频时间长度是否存在
        is_variable_len_video = exists(video_time_len)

        if is_variable_len_video:
            # 确保视频时间长度有效
            assert ((video_time_len > 0) & (video_time_len <= time_seq_len)).all(), '`video_time_len` has invalid time lengths'
            # 生成时间掩码，形状为 (batch_size, time_seq_len)
            time_mask = lens_to_mask(video_time_len, time_seq_len)

        # handle actions, but allow for state dynamics model to be trained independently
        # 处理动作，但允许独立训练状态动态模型

        # when training, adding action embedding depends on the condition dropout probability 
        # 在训练时，添加动作嵌入取决于条件 dropout 概率

        add_action_embed = (
            exists(actions) and
            (not self.training or random() >= self.cfg_train_action_dropout) # 如果在训练时，添加动作嵌入的概率为 1 - cfg_train_action_dropout
        )

        if add_action_embed:
            # 确保动作张量的维度为 2 或 3
            assert actions.ndim in {2, 3} # either Int[b, n] or Int[b, n, a] -> for multiple keys being pressed
            # 确保动作张量的时间步数与 time_seq_len 相同
            assert actions.shape[1] == time_seq_len

            # 判断是否为多动作
            is_multi_action = actions.ndim == 3

            if is_multi_action and sort_actions:
                # 将负数填充为 1e6
                actions = actions.masked_fill(actions < 0, 1e6)
                # 对动作进行排序
                actions = actions.sort(dim = -1).values
                # 将排序后的 1e6 填充回负数
                actions = actions.masked_fill(actions == 1e6, -1)

            # 确保定义了动作嵌入
            assert exists(self.action_embed), '`num_actions` must be defined for action embedding on Genie2 before dynamics model can be conditioned on actions'

            # 将动作张量打包为 (batch_size, n, *)
            actions, _ = pack_one(actions, 'b n *')

            # 找到没有动作的位置
            no_actions = actions < 0
            # 将没有动作的位置填充为 0
            actions = actions.masked_fill(no_actions, 0)

            # 对动作进行嵌入
            action_embed = self.action_embed(actions)
            # 将没有动作的位置的嵌入设置为 0
            action_embed = einx.where('b n a, b n a d, -> b n a d', ~no_actions, action_embed, 0.)

            # 对动作嵌入进行求和归约
            action_embed = reduce(action_embed, 'b n a d -> b n d', 'sum')

        # encode the state, if state codes are given during sampling, fetch the codes from the vq codebook
        # 编码状态，如果提供了 state，则编码 state；如果提供了 state_codes，则从 vq 代码本中获取 codes

        if exists(state):
            # 对状态进行编码，得到量化后的潜在代码、潜在代码索引和承诺损失
            quantized_latents, latent_indices, commit_loss = self.encode_state(state)

        elif exists(state_codes):
            # 如果提供了 state_codes，则使用 state_codes 作为潜在代码索引
            latent_indices = state_codes
            # 从 vq 代码本中获取量化后的潜在代码
            quantized_latents = self.vq.get_codes_from_indices(latent_indices)

        # handle rotary positions
        # 处理旋转位置

        # repeat time across space
        # 将时间序列重复到空间维度

        # 获取潜在序列的长度
        latent_seq_len = quantized_latents.shape[-2]
        # 计算空间重复因子，确保时间序列长度覆盖潜在序列
        spatial_repeat_factor = ceil(latent_seq_len / time_seq_len)

        # 将时间序列重复到空间维度，形状变为 (n * r)
        time_seq = repeat(time_seq, 'n -> (n r)', r = spatial_repeat_factor)

        # give meta tokens position of -1
        # 给元标记的位置赋值为 -1

        # 在时间序列前填充元标记的位置，填充值为 -1
        time_seq = F.pad(time_seq, (self.num_meta_tokens, 0), value = -1)

        if add_action_embed:
            # 如果添加了动作嵌入，则将动作嵌入重复到空间维度
            action_embed = repeat(action_embed, 'b n d-> b (n r) d', r = spatial_repeat_factor)

        # 生成时间旋转位置嵌入
        time_rotary_pos = self.time_rotary(time_seq)

        # if returning loss, setup labels for autoregressive loss
        # 如果需要返回损失，则设置自回归损失的标签

        if return_loss:
            # 去除最后一个时间步的潜在代码，用于自回归训练
            quantized_latents = quantized_latents[:, :-1]

            # 解包旋转位置嵌入
            rotary_pos, xpos_scale = time_rotary_pos
            # 去除最后一个时间步的旋转位置
            time_rotary_pos = (rotary_pos[:, :-1], xpos_scale)

            if is_variable_len_video:
                # 如果视频长度可变，则重复时间掩码到空间维度
                time_mask = repeat(time_mask, 'b n -> b (n r)', r = spatial_repeat_factor)
                # 使用时间掩码填充潜在代码索引为 -1
                latent_indices = latent_indices.masked_fill(time_mask, -1)

            # 设置标签为下一个时间步的潜在代码索引
            labels = latent_indices[:, 1:]

            # 初始化动作标签
            action_labels = None

            if add_action_embed:
                # 去除最后一个时间步的动作嵌入
                action_embed = action_embed[:, :-1]
                # 设置动作标签为下一个时间步的动作
                action_labels = actions[:, 1:]

        # project in
        # 投影到模型维度
        
        # 将量化后的潜在代码投影到模型维度
        tokens = self.latent_to_model(quantized_latents)

        # 获取 tokens 的序列长度
        tokens_seq_len = tokens.shape[-2]

        # add action conditioning, if needed
        # 如果需要，添加动作条件

        if add_action_embed:
            # 截取动作嵌入到与 tokens 相同的序列长度
            action_embed = action_embed[:, :tokens_seq_len]
            # 将动作嵌入添加到 tokens 中，实现动作条件
            tokens = tokens + action_embed

        # autoregressive attention
        # 自回归注意力

        embed = self.transformer(
            tokens, # 输入 tokens
            rotary_pos_emb = time_rotary_pos # 传入时间旋转位置嵌入
        )

        # project out
        # 投影输出

        # 将 Transformer 的输出投影回潜在空间
        tokens = self.model_to_latent(embed)

        # cross entropy loss off the vq codebook
        # 计算交叉熵损失

        # 获取向量量化器的码本
        codebook = self.vq.codebook

        # 计算 tokens 与码本之间的距离，并取负数作为 logits
        logits = -torch.cdist(tokens, codebook)

        if not return_loss:
            # 如果不需要返回损失，则直接返回 logits
            return logits
        
        # 计算自回归损失
        state_autoregressive_loss = F.cross_entropy(
            rearrange(logits, 'b n l -> b l n'), # 重塑 logits 形状为 (batch_size, sequence_length, num_embeddings)
            labels, # 标签为下一个时间步的潜在代码索引
            ignore_index = -1 # 忽略填充位置的损失
        ) 

        # 计算总损失
        total_loss = (
            state_autoregressive_loss + # 自回归损失
            commit_loss * self.vq_commit_loss_weight # 承诺损失乘以承诺损失权重
        )

        # maybe action loss
        # 可能的动作损失

        # 初始化动作损失为 0
        action_loss = self.zero

        # 如果需要计算动作损失
        if self.has_action_loss:
            # 判断是否为单个动作
            is_single_action = actions.ndim == 2 or actions.shape[-1] == 1

            if not self.allow_multiple_actions:
                # 如果不允许多个动作，则断言为单个动作
                assert is_single_action, 'you need to set `allow_multiple_actions = True` on init to learn and decode multiple actions'

            # 计算动作的时间长度
            action_time_len = tokens_seq_len // spatial_repeat_factor
            # 根据空间重复因子计算向下取整的动作时间长度
            round_down_by_space_len = action_time_len * spatial_repeat_factor
            # 对 Transformer 输出进行平均池化，得到动作嵌入
            action_embed = reduce(embed[:, :round_down_by_space_len], 'b (t s) d -> b t d', 'mean', t = action_time_len)

            if is_single_action:
                # 使用动作预测层计算单个动作的 logits
                action_logits = self.to_action_pred(action_embed)

                if actions.ndim == 3:
                    # 如果动作维度为 3，则重塑为 2 维
                    actions = rearrange(actions, '... 1 -> ...')

                # 设置动作标签为下一个时间步的动作
                action_labels = actions[:, 1:]

            else:
                # 将动作张量打包为 (batch_size, n, *)
                actions, _ = pack_one(actions, 'b n *')
                # 获取输入动作的数量
                inp_num_actions = actions.shape[-1]
                # 确保输入动作数量不超过最大动作数量
                assert inp_num_actions <= self.max_num_actions, f'maximum number of actions is set at {self.max_num_actions}'

                # 重塑动作嵌入为 (batch_size * time_steps, 1, d)
                action_embed = rearrange(action_embed, 'b t d -> (b t) 1 d')
                # 重复动作位置嵌入
                action_pos_embed = repeat(self.action_pos_embed[:inp_num_actions], 'a d -> bt a d', bt = action_embed.shape[0])

                # 将动作嵌入和动作位置嵌入拼接
                action_embed = torch.cat((action_embed, action_pos_embed), dim = -2)
                # 使用动作预测层计算动作 logits
                action_logits = self.to_action_pred(action_embed)

                # prepare the action labels, adding the action end token appropriately
                # 准备动作标签，添加动作结束符

                # 设置动作标签为下一个时间步的动作
                action_labels = actions[:, 1:]
                # 在动作标签的最后一个维度填充动作结束符
                action_labels = F.pad(action_labels, (0, 1), value = -1)
                # 计算每个时间步的动作数量
                num_actions_per_time = (action_labels >= 0).sum(dim = -1, keepdim = True)
                # 在每个时间步的动作数量位置添加动作结束符
                action_labels = action_labels.scatter(-1, num_actions_per_time, self.action_eos_id)

                # handle variable lengthed videos
                # 处理可变长度的视频

                if is_variable_len_video:
                    # 如果视频长度可变，则使用时间掩码填充动作标签为 -1
                    action_labels = action_labels.masked_fill(time_mask[:, :-1], -1)

                # fold time into batch
                # 将时间维度折叠到批次维度

                # 重塑动作标签为 (batch_size * time_steps, a)
                action_labels = rearrange(action_labels, 'b t a -> (b t) a')

            # cross entropy loss for predicted action on the action transformer head (hierarchical transformer)
            # 计算动作交叉熵损失

            action_loss = F.cross_entropy(
                # 重塑 logits 为 (batch_size, sequence_length, num_actions)
                rearrange(action_logits, 'b n l -> b l n'),
                action_labels, # 动作标签
                ignore_index = -1 # 忽略填充位置的损失
            )

            total_loss = (
                total_loss + # 总损失
                action_loss * self.action_autoregressive_loss_weight # 动作自回归损失乘以权重
            )

        # 返回总损失和各个损失项
        return total_loss, (state_autoregressive_loss, commit_loss, action_loss)
