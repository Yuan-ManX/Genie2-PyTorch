import torch
from genie2 import Genie2


# 创建 Genie2 模型实例
genie = Genie2(
    dim = 512, # 模型的基本维度
    depth = 12, # Transformer 的深度（层数）
    dim_latent = 768, # 潜在空间的维度
    num_actions = 256, # 动作的数量
    latent_channel_first = True, # 潜在张量是否以通道优先的形式存储
    is_video_enc_dec = True # 是否使用视频编码器和解码器
)

# 生成随机的视频张量，形状为 (batch_size, channels, time_steps, height, width)
video = torch.randn(2, 768, 3, 2, 2)

# 生成随机的动作张量，形状为 (batch_size, time_steps)
actions = torch.randint(0, 256, (2, 3))

# 前向传播，计算损失
# 调用 Genie2 模型的前向传播方法，输入视频和动作，返回损失和损失分解
loss, breakdown = genie(video, actions = actions) 

# 反向传播，计算梯度
loss.backward()

# 生成视频
# 使用 Genie2 模型的 generate 方法生成视频，输入初始帧和帧数
generated_video = genie.generate(video[:, :, 0], num_frames = 16)

# 生成的视频张量形状为 (2, 768, 17, 2, 2)
assert generated_video.shape == (2, 768, 16 + 1, 2, 2)


# Interactive
# 交互式生成模式

# 创建 Genie2 模型实例
genie = Genie2(
    dim = 512, # 模型的基本维度
    depth = 12, # Transformer 的深度（层数）
    dim_latent = 768, # 潜在空间的维度
    num_actions = 256, # 动作的数量
    latent_channel_first = True, # 潜在张量是否以通道优先的形式存储
    is_video_enc_dec = True # 是否使用视频编码器和解码器
)

# 生成随机的视频张量，形状为 (batch_size, channels, time_steps, height, width)
video = torch.randn(1, 768, 3, 2, 2)

# 生成随机的动作张量，形状为 (batch_size, time_steps)
actions = torch.randint(0, 256, (1, 3))

# 前向传播，计算损失
# 调用 Genie2 模型的前向传播方法，输入视频和动作，返回损失和损失分解
loss, breakdown = genie(video, actions = actions)

# 反向传播，计算梯度
loss.backward()

# 生成视频并进入交互式模式
generated_video, actions = genie.generate(
    video[:, :, 0], # 输入初始帧，形状为 (1, 768, 2, 2)
    num_frames = 16, # 生成 16 帧视频
    interactive = True, # 启用交互式模式
    init_action = 0 # 设置初始动作
)

# you will be prompted to enter the next action id(s) at every next time frame of the video
# 在每个视频时间帧的下一步，系统将提示您输入下一个动作 ID

# 生成的视频张量形状为 (1, 768, 17, 2, 2)
assert generated_video.shape == (1, 768, 16 + 1, 2, 2)
