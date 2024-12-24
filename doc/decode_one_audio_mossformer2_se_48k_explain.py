
def decode_one_audio_mossformer2_se_48k(model, device, inputs, args):
    """
        This function decodes audio input using the following steps:
        该函数使用以下步骤解码音频输入：
        1. Normalizes the audio input to a maximum WAV value.
        1. 将音频输入归一化到最大 WAV 值。
        2. Checks the length of the input to decide between online decoding and batch processing.
        2. 检查输入的长度以决定是在线解码还是批处理。
        3. For longer inputs, processes the audio in segments using a sliding window.
        3. 对于较长的输入，使用滑动窗口将音频分段处理。
        4. Computes filter banks and their deltas for the audio segment.
        4. 计算音频段的滤波器组及其增量。
        5. Passes the filter banks through the model to get a predicted mask.
        5. 将滤波器组传递给模型以获取预测掩码。
        6. Applies the mask to the spectrogram of the audio segment and reconstructs the audio.
        6. 将掩码应用于音频段的频谱图并重建音频。
        7. For shorter inputs, processes them in one go without segmentation.
        7. 对于较短的输入，一次性处理而不进行分段。

        Args:
            model (nn.Module): The trained MossFormer2 model used for decoding.
            model (nn.Module): 用于解码的训练好的 MossFormer2 模型。
            device (torch.device): The device (CPU or GPU) for computation.
            device (torch.device): 用于计算的设备（CPU 或 GPU）。
            inputs (torch.Tensor): Input audio tensor of shape (B, T), where B is the batch size and T is the number of time steps.
            inputs (torch.Tensor): 输入音频张量，形状为 (B, T)，其中 B 是批量大小，T 是时间步数。
            args (Namespace): Contains arguments for sampling rate, window size, and other parameters.
            args (Namespace): 包含采样率、窗口大小和其他参数的参数。

        Returns:
            numpy.ndarray: The decoded audio output, normalized to the range [-1, 1].
            numpy.ndarray: 解码后的音频输出，归一化到 [-1, 1] 范围。
    """

# 从输入张量中提取第一个元素
inputs = inputs[0, :]  # Extract the first element from the input tensor

# 获取输入音频的长度
input_len = inputs.shape[0]  # Get the length of the input audio

# 将输入归一化到最大 WAV 值
inputs = inputs * MAX_WAV_VALUE  # Normalize the input to the maximum WAV value

# 检查输入长度是否超过在线解码的定义阈值
if input_len > args.sampling_rate * args.one_time_decode_length:  # 20秒
    # 设置在线解码为 True
    online_decoding = True
    if online_decoding:
        # 定义窗口长度（例如，48kHz 时为 4 秒）
        window = int(args.sampling_rate * args.decode_window)  # Define window length (e.g., 4s for 48kHz)
        # 定义步长（例如，48kHz 时为 3 秒）
        stride = int(window * 0.75)  # Define stride length (e.g., 3s for 48kHz)
        # 更新填充后的长度
        t = inputs.shape[0]  # Update length after potential padding

        # 如果需要，填充输入以匹配窗口大小
        if t < window:
            # 填充输入
            inputs = np.concatenate([inputs, np.zeros(window - t)], 0)
        elif t < window + stride:
            # 计算填充长度
            padding = window + stride - t
            # 填充输入
            inputs = np.concatenate([inputs, np.zeros(padding)], 0)
        else:
            if (t - window) % stride != 0:
                # 计算填充长度
                padding = t - (t - window) // stride * stride
                # 填充输入
                inputs = np.concatenate([inputs, np.zeros(padding)], 0)

        # 转换为 Torch 张量
        """
            举例：
            # 假设我们最初的输入是一个 NumPy 数组，比如:
            inputs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
            # 这行代码会把 NumPy 数组转成 PyTorch 张量，数据类型为 FloatTensor。
            audio = torch.from_numpy(inputs).type(torch.FloatTensor)
            # 转换后，audio 就是一个包含相同数值的张量:
            # tensor([0.1000, 0.2000, 0.3000, 0.4000, 0.5000])
        """
        audio = torch.from_numpy(inputs).type(torch.FloatTensor)  # Convert to Torch tensor
        # 更新转换后的长度
        t = audio.shape[0]  # Update length after conversion
        # 初始化输出张量
        outputs = torch.from_numpy(np.zeros(t))  # Initialize output tensor
        # 确定边缘忽略的长度//用于做滑动窗口计算//滑动窗口有单独的文档解释
        give_up_length = (window - stride) // 2  # Determine length to ignore at the edges
        # 潜在内存长度的占位符
        dfsmn_memory_length = 0  # Placeholder for potential memory length
        # 初始化滑动窗口的当前索引
        current_idx = 0  # Initialize current index for sliding window

        # 在滑动窗口段中处理音频
        while current_idx + window <= t:
            # 选择适当的音频段进行处理
            if current_idx < dfsmn_memory_length:
                audio_segment = audio[0:current_idx + window]
            else:
                audio_segment = audio[current_idx - dfsmn_memory_length:current_idx + window]

            # 计算音频段的滤波器组//滤波器组有单独的文档解释，函数详见 utils.misc.compute_fbank
            fbanks = compute_fbank(audio_segment.unsqueeze(0), args)
            
            # 计算滤波器组的增量
            # 转置以计算增量
            fbank_tr = torch.transpose(fbanks, 0, 1)  # Transpose for delta computation
            # 一阶增量
            fbank_delta = torchaudio.functional.compute_deltas(fbank_tr)  # First-order delta
            # 二阶增量
            fbank_delta_delta = torchaudio.functional.compute_deltas(fbank_delta)  # Second-order delta
            
            # 转置回原始形状
            fbank_delta = torch.transpose(fbank_delta, 0, 1)
            fbank_delta_delta = torch.transpose(fbank_delta_delta, 0, 1)

            # 将原始滤波器组与其增量连接
            fbanks = torch.cat([fbanks, fbank_delta, fbank_delta_delta], dim=1)
            # 添加批次维度并移动到设备
            fbanks = fbanks.unsqueeze(0).to(device)  # Add batch dimension and move to device

            # 将滤波器组传递给模型
            Out_List = model(fbanks)
            # 从输出中获取预测掩码
            pred_mask = Out_List[-1]  # Get the predicted mask from the output

            # 对音频段应用 STFT
            spectrum = stft(audio_segment, args)
            # 维度置换以进行掩码
            pred_mask = pred_mask.permute(2, 1, 0)  # Permute dimensions for masking
            # 将掩码应用于频谱
            masked_spec = spectrum.cpu() * pred_mask.detach().cpu()  # Apply mask to the spectrum
            # 转换为复数形式
            masked_spec_complex = masked_spec[:, :, 0] + 1j * masked_spec[:, :, 1]  # Convert to complex form

            # 从掩码频谱图中重建音频
            output_segment = istft(masked_spec_complex, args, len(audio_segment))

            # 将输出段存储在输出张量中
            if current_idx == 0:
                outputs[current_idx:current_idx + window - give_up_length] = output_segment[:-give_up_length]
            else:
                # 获取最新的窗口输出
                output_segment = output_segment[-window:]  # Get the latest window of output
                outputs[current_idx + give_up_length:current_idx + window - give_up_length] = output_segment[give_up_length:-give_up_length]
            
            # 移动到下一个段
            current_idx += stride  # Move to the next segment

else:
    # 如果音频长度小于阈值，则一次性处理整个音频
    audio = torch.from_numpy(inputs).type(torch.FloatTensor)
    # 计算音频段的滤波器组；具体见：utils.misc.compute_fbank
    fbanks = compute_fbank(audio.unsqueeze(0), args)

    # 计算滤波器组的增量
    # 转置以计算增量
    fbank_tr = torch.transpose(fbanks, 0, 1)
    # 一阶增量
    fbank_delta = torchaudio.functional.compute_deltas(fbank_tr)
    # 二阶增量
    fbank_delta_delta = torchaudio.functional.compute_deltas(fbank_delta)
    # 转置回原始形状
    fbank_delta = torch.transpose(fbank_delta, 0, 1)
    fbank_delta_delta = torch.transpose(fbank_delta_delta, 0, 1)

    # 将原始滤波器组与其增量连接
    fbanks = torch.cat([fbanks, fbank_delta, fbank_delta_delta], dim=1)
    # 添加批次维度并移动到设备
    fbanks = fbanks.unsqueeze(0).to(device)  # Add batch dimension and move to device

    # 将滤波器组传递给模型
    Out_List = model(fbanks)
    # 获取预测掩码
    pred_mask = Out_List[-1]  # Get the predicted mask
    # 对音频应用 STFT
    spectrum = stft(audio, args)  # Apply STFT to the audio
    # 维度置换以进行掩码
    pred_mask = pred_mask.permute(2, 1, 0)  # Permute dimensions for masking
    # 将掩码应用于频谱
    masked_spec = spectrum * pred_mask.detach().cpu()  # Apply mask to the spectrum
    # 转换为复数形式
    masked_spec_complex = masked_spec[:, :, 0] + 1j * masked_spec[:, :, 1]  # Convert to complex form
    
    # 从掩码频谱图中重建音频
    outputs = istft(masked_spec_complex, args, len(audio))

# 返回归一化到 [-1, 1] 的输出
return outputs.numpy() / MAX_WAV_VALUE  # Return the output normalized to [-1, 1]