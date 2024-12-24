
#from networks.py stratup to line 266  
"""
作者: Shengkui Zhao, Zexu Pan
"""

import torch  # 导入 PyTorch 库
import soundfile as sf  # 导入 soundfile 库，用于音频文件的读写
import os  # 导入 os 库，用于操作系统相关功能
import subprocess  # 导入 subprocess 库，用于执行系统命令
from tqdm import tqdm  # 导入 tqdm 库，用于显示进度条
from utils.decode import decode_one_audio  # 从 utils.decode 模块导入 decode_one_audio 函数
from dataloader.dataloader import DataReader  # 从 dataloader.dataloader 模块导入 DataReader 类

class SpeechModel:
    """
    SpeechModel 类是一个基类，设计用于处理语音处理任务，
    如加载、处理和解码音频数据。它初始化计算设备（CPU 或 GPU），
    并持有与模型相关的属性。该类是灵活的，旨在由特定的语音模型扩展，
    用于语音增强、语音分离、目标说话人提取等任务。

    属性:
    - args: 包含配置设置的参数解析对象。
    - device: 模型运行的设备（CPU 或 GPU）。
    - model: 用于语音处理任务的实际模型（由子类加载）。
    - name: 模型名称的占位符。
    - data: 用于存储与模型相关的任何附加数据的字典，如音频输入。
    """

    def __init__(self, args):
        """
        初始化 SpeechModel 类，确定计算设备（GPU 或 CPU）以运行模型，
        基于系统的可用性。

        参数:
        - args: 包含是否使用 CUDA（GPU）等设置的参数解析对象。
        """
        # 检查是否有可用的 GPU
        if torch.cuda.is_available():
            # 使用自定义方法查找具有最多空闲内存的 GPU
            free_gpu_id = self.get_free_gpu()
            if free_gpu_id is not None:
                args.use_cuda = 1
                torch.cuda.set_device(free_gpu_id)
                self.device = torch.device('cuda')
            else:
                # 如果没有检测到 GPU，则使用 CPU
                args.use_cuda = 0
                self.device = torch.device('cpu')
        else:
            # 如果没有检测到 GPU，则使用 CPU
            args.use_cuda = 0
            self.device = torch.device('cpu')

        self.args = args  # 保存参数解析对象
        self.model = None  # 初始化模型为 None
        self.name = None  # 初始化模型名称为 None
        self.data = {}  # 初始化数据字典为空

    def get_free_gpu(self):
        """
        使用 'nvidia-smi' 命令识别具有最多空闲内存的 GPU 并返回其索引。

        该函数查询系统上的可用 GPU，并确定哪个 GPU 具有最多的空闲内存。
        它使用 `nvidia-smi` 命令行工具来收集 GPU 内存使用数据。如果成功，
        它返回具有最多空闲内存的 GPU 的索引。如果查询失败或发生错误，
        它返回 None。

        返回:
        int: 具有最多空闲内存的 GPU 的索引，如果没有找到 GPU 或发生错误，则返回 None。
        """
        try:
            # 运行 nvidia-smi 以查询 GPU 内存使用情况和空闲内存
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
            gpu_info = result.stdout.decode('utf-8').strip().split('\n')

            free_gpu = None
            max_free_memory = 0
            for i, info in enumerate(gpu_info):
                used, free = map(int, info.split(','))
                if free > max_free_memory:
                    max_free_memory = free
                    free_gpu = i
            return free_gpu
        except Exception as e:
            print(f"查找空闲 GPU 时出错: {e}")
            return None

    def download_model(self, model_name):
        """
        下载模型检查点到指定目录。

        参数:
        - model_name: 模型名称。
        """
        checkpoint_dir = self.args.checkpoint_dir
        from huggingface_hub import snapshot_download
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print(f'正在下载 {model_name} 的检查点')
        try:
            snapshot_download(repo_id=f'alibabasglab/{model_name}', local_dir=checkpoint_dir)
            return True
        except:
            return False
            
    def load_model(self):
        """
        从指定目录加载预训练模型检查点。它检查最佳模型（'last_best_checkpoint'）
        或最新的检查点（'last_checkpoint'）。如果找到模型，它将模型状态加载到
        当前模型实例中。

        如果没有找到检查点，它会打印警告信息。

        步骤:
        - 搜索最佳模型检查点或最新的检查点。
        - 从检查点文件加载模型的状态字典。

        异常:
        - FileNotFoundError: 如果没有找到 'last_best_checkpoint' 或 'last_checkpoint' 文件。
        """
        # 定义最佳模型和最新检查点的路径
        best_name = os.path.join(self.args.checkpoint_dir, 'last_best_checkpoint')
        # 检查是否存在最新的最佳检查点
        if not os.path.isfile(best_name):
            if not self.download_model(self.name):
                # 如果下载不成功
                print(f'警告: 下载模型 {self.name} 不成功。请重试或手动从 https://huggingface.co/alibabasglab/{self.name}/tree/main 下载！')
                return

        # 从文件中读取模型的检查点名称
        with open(best_name, 'r') as f:
            model_name = f.readline().strip()
        
        # 形成模型检查点的完整路径
        checkpoint_path = os.path.join(self.args.checkpoint_dir, model_name)
        
        # 将检查点文件加载到内存中（map_location 确保与不同设备兼容）
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

        # 将模型的状态字典（权重和偏差）加载到当前模型中
        if 'model' in checkpoint:
            pretrained_model = checkpoint['model']
        else:
            pretrained_model = checkpoint
        state = self.model.state_dict()
        for key in state.keys():
            if key in pretrained_model and state[key].shape == pretrained_model[key].shape:
                state[key] = pretrained_model[key]
            elif key.replace('module.', '') in pretrained_model and state[key].shape == pretrained_model[key.replace('module.', '')].shape:
                 state[key] = pretrained_model[key.replace('module.', '')]
            elif 'module.'+key in pretrained_model and state[key].shape == pretrained_model['module.'+key].shape:
                 state[key] = pretrained_model['module.'+key]
            elif self.print: print(f'{key} 未加载')
        self.model.load_state_dict(state)
        #print(f'成功加载 {model_name} 进行解码')

    def decode(self):
        """
        使用加载的模型解码输入音频数据，并确保输出与原始音频长度匹配。

        该方法通过语音模型（例如用于增强、分离等）处理音频，
        并将结果音频截断以匹配原始输入的长度。该方法支持多说话者音频，
        如果模型处理多说话者音频。

        返回:
        output_audio: 处理后的解码音频，截断到输入音频长度。
                      如果处理多说话者音频，返回每个说话者的截断音频列表。
        """
        # 使用加载的模型在指定设备上解码音频（例如 CPU 或 GPU）
        output_audio = decode_one_audio(self.model, self.device, self.data['audio'], self.args)

        # 确保解码后的输出与输入音频的长度匹配
        if isinstance(output_audio, list):
            # 如果是多说话者音频（输出列表），则截断每个说话者的音频到输入长度
            for spk in range(self.args.num_spks):
                output_audio[spk] = output_audio[spk][:self.data['audio_len']]
        else:
            # 单一输出，截断到输入音频长度
            output_audio = output_audio[:self.data['audio_len']]
    
        return output_audio

    def process(self, input_path, online_write=False, output_path=None):
        """
        从指定的输入路径加载和处理音频文件。可选地，
        将输出音频文件写入指定的输出目录。
        
        参数:
            input_path (str): 输入音频文件或文件夹的路径。
            online_write (bool): 是否实时写入处理后的音频到磁盘。
            output_path (str): 可选的输出文件路径。如果为 None，
                               输出将存储在 self.result 中。
        
        返回:
            dict 或 ndarray: 处理后的音频结果，作为字典或单个数组，
                             取决于处理的音频文件数量。
                             如果启用了 online_write，则返回 None。
        """
        
        self.result = {}  # 初始化结果字典
        self.args.input_path = input_path  # 设置输入路径
        data_reader = DataReader(self.args)  # 初始化数据读取器以加载音频文件

        # 检查是否启用了在线写入
        if online_write:
            output_wave_dir = self.args.output_dir  # 设置默认输出目录
            if isinstance(output_path, str):  # 如果提供了特定的输出路径，则使用它
                output_wave_dir = os.path.join(output_path, self.name)
            # 如果输出目录不存在，则创建它
            if not os.path.isdir(output_wave_dir):
                os.makedirs(output_wave_dir)
        
        num_samples = len(data_reader)  # 获取要处理的样本总数
        print(f'正在运行 {self.name} ...')  # 显示正在使用的模型

        if self.args.task == 'target_speaker_extraction':
            from utils.video_process import process_tse
            assert online_write == True
            process_tse(self.args, self.model, self.device, data_reader, output_wave_dir)
        else:
            # 禁用梯度计算以提高推理效率
            with torch.no_grad():
                for idx in tqdm(range(num_samples)):  # 循环处理所有音频样本
                    self.data = {}
                    # 从数据读取器中读取音频、波形 ID 和音频长度
                    input_audio, wav_id, input_len, scalar = data_reader[idx]
                    # 将输入音频和元数据存储在 self.data 中
                    self.data['audio'] = input_audio
                    self.data['id'] = wav_id
                    self.data['audio_len'] = input_len
                    
                    # 执行音频解码/处理
                    output_audio = self.decode()

                    # 执行音频重新归一化
                    if not isinstance(output_audio, list):
                        output_audio = output_audio * scalar
                        
                    if online_write:
                        # 如果启用了在线写入，将输出音频保存到文件
                        if isinstance(output_audio, list):
                            # 如果是多说话者输出，分别保存每个说话者的输出
                            for spk in range(self.args.num_spks):
                                output_file = os.path.join(output_wave_dir, wav_id.replace('.wav', f'_s{spk+1}.wav'))
                                sf.write(output_file, output_audio[spk], self.args.sampling_rate)
                        else:
                            # 单一说话者或标准输出
                            output_file = os.path.join(output_wave_dir, wav_id)
                            sf.write(output_file, output_audio, self.args.sampling_rate)
                    else:
                        # 如果不写入磁盘，将输出存储在结果字典中
                        self.result[wav_id] = output_audio
            
            # 如果不写入磁盘，则返回处理结果
            if not online_write:
                if len(self.result) == 1:
                    # 如果只有一个结果，直接返回它
                    return next(iter(self.result.values()))
                else:
                    # 否则，返回整个结果字典
                    return self.result

#from networks.py line 267  to end  
def write(self, output_path, add_subdir=False, use_key=False):
        """
        将处理后的音频结果写入指定的输出路径。

        参数:
            output_path (str): 保存处理后音频的目录或文件路径。如果未提供，默认为 self.args.output_dir。
            add_subdir (bool): 如果为 True，则在输出路径中添加模型名称作为子目录。
            use_key (bool): 如果为 True，则使用结果字典的键（音频文件 ID）作为文件名。

        返回:
            None: 输出写入磁盘，不返回数据。
        """

        # 确保输出路径是字符串。如果未提供，使用默认输出目录
        if not isinstance(output_path, str):
            output_path = self.args.output_dir

        # 如果启用了 add_subdir，则为模型名称创建子目录
        if add_subdir:
            if os.path.isfile(output_path):
                print(f'文件已存在: {output_path}，请删除后重试！')
                return
            output_path = os.path.join(output_path, self.name)
            if not os.path.isdir(output_path):
                os.makedirs(output_path)

        # 使用键作为文件名时，确保正确的目录设置
        if use_key and not os.path.isdir(output_path):
            if os.path.exists(output_path):
                print(f'文件已存在: {output_path}，请删除后重试！')
                return
            os.makedirs(output_path)
        # 如果不使用键且输出路径是目录，检查是否有冲突
        if not use_key and os.path.isdir(output_path):
            print(f'目录已存在: {output_path}，请删除后重试！')
            return

        # 遍历结果字典，将处理后的音频写入磁盘
        for key in self.result:
            if use_key:
                # 如果使用键，基于结果字典的键（音频 ID）格式化文件名
                if isinstance(self.result[key], list):  # 对于多说话者输出
                    for spk in range(self.args.num_spks):
                        sf.write(os.path.join(output_path, key.replace('.wav', f'_s{spk+1}.wav')),
                                 self.result[key][spk], self.args.sampling_rate)
                else:
                    sf.write(os.path.join(output_path, key), self.result[key], self.args.sampling_rate)
            else:
                # 如果不使用键，直接将音频写入指定的输出路径
                if isinstance(self.result[key], list):  # 对于多说话者输出
                    for spk in range(self.args.num_spks):
                        sf.write(output_path.replace('.wav', f'_s{spk+1}.wav'),
                                 self.result[key][spk], self.args.sampling_rate)
                else:
                    sf.write(output_path, self.result[key], self.args.sampling_rate)

# 特定子任务的模型类

class CLS_FRCRN_SE_16K(SpeechModel):
    """
    SpeechModel 的子类，使用 FRCRN 架构实现 16 kHz 语音增强模型。
    
    参数:
        args (Namespace): 包含模型配置和路径的参数解析对象。
    """

    def __init__(self, args):
        # 初始化父类 SpeechModel
        super(CLS_FRCRN_SE_16K, self).__init__(args)
        
        # 导入 FRCRN 语音增强模型用于 16 kHz
        from models.frcrn_se.frcrn import FRCRN_SE_16K
        
        # 初始化模型
        self.model = FRCRN_SE_16K(args).model
        self.name = 'FRCRN_SE_16K'
        
        # 加载预训练模型检查点
        self.load_model()
        
        # 将模型移动到适当的设备（GPU/CPU）
        if args.use_cuda == 1:
            self.model.to(self.device)
        
        # 将模型设置为评估模式（不计算梯度）
        self.model.eval()

class CLS_MossFormer2_SE_48K(SpeechModel):
    """
    SpeechModel 的子类，使用 MossFormer2 架构实现 48 kHz 语音增强模型。
    
    参数:
        args (Namespace): 包含模型配置和路径的参数解析对象。
    """

    def __init__(self, args):
        # 初始化父类 SpeechModel
        super(CLS_MossFormer2_SE_48K, self).__init__(args)
        
        # 导入 MossFormer2 语音增强模型用于 48 kHz
        from models.mossformer2_se.mossformer2_se_wrapper import MossFormer2_SE_48K
        
        # 初始化模型
        self.model = MossFormer2_SE_48K(args).model
        self.name = 'MossFormer2_SE_48K'
        
        # 加载预训练模型检查点
        self.load_model()
        
        # 将模型移动到适当的设备（GPU/CPU）
        if args.use_cuda == 1:
            self.model.to(self.device)
        
        # 将模型设置为评估模式（不计算梯度）
        self.model.eval()

class CLS_MossFormerGAN_SE_16K(SpeechModel):
    """
    SpeechModel 的子类，使用 MossFormerGAN 架构实现 16 kHz 语音增强，
    利用 GAN 进行语音处理。
    
    参数:
        args (Namespace): 包含模型配置和路径的参数解析对象。
    """

    def __init__(self, args):
        # 初始化父类 SpeechModel
        super(CLS_MossFormerGAN_SE_16K, self).__init__(args)
        
        # 导入 MossFormerGAN 语音增强模型用于 16 kHz
        from models.mossformer_gan_se.generator import MossFormerGAN_SE_16K
        
        # 初始化模型
        self.model = MossFormerGAN_SE_16K(args).model
        self.name = 'MossFormerGAN_SE_16K'
        
        # 加载预训练模型检查点
        self.load_model()
        
        # 将模型移动到适当的设备（GPU/CPU）
    def write(self, output_path, add_subdir=False, use_key=False):
        """
        将处理后的音频结果写入指定的输出路径。

        参数:
            output_path (str): 保存处理后音频的目录或文件路径。如果未提供，默认为 self.args.output_dir。
            add_subdir (bool): 如果为 True，则在输出路径中添加模型名称作为子目录。
            use_key (bool): 如果为 True，则使用结果字典的键（音频文件 ID）作为文件名。

        返回:
            None: 输出写入磁盘，不返回数据。
        """

        # 确保输出路径是字符串。如果未提供，使用默认输出目录
        if not isinstance(output_path, str):
            output_path = self.args.output_dir

        # 如果启用了 add_subdir，则为模型名称创建子目录
        if add_subdir:
            if os.path.isfile(output_path):
                print(f'文件已存在: {output_path}，请删除后重试！')
                return
            output_path = os.path.join(output_path, self.name)
            if not os.path.isdir(output_path):
                os.makedirs(output_path)

        # 使用键作为文件名时，确保正确的目录设置
        if use_key and not os.path.isdir(output_path):
            if os.path.exists(output_path):
                print(f'文件已存在: {output_path}，请删除后重试！')
                return
            os.makedirs(output_path)
        # 如果不使用键且输出路径是目录，检查是否有冲突
        if not use_key and os.path.isdir(output_path):
            print(f'目录已存在: {output_path}，请删除后重试！')
            return

        # 遍历结果字典，将处理后的音频写入磁盘
        for key in self.result:
            if use_key:
                # 如果使用键，基于结果字典的键（音频 ID）格式化文件名
                if isinstance(self.result[key], list):  # 对于多说话者输出
                    for spk in range(self.args.num_spks):
                        sf.write(os.path.join(output_path, key.replace('.wav', f'_s{spk+1}.wav')),
                                 self.result[key][spk], self.args.sampling_rate)
                else:
                    sf.write(os.path.join(output_path, key), self.result[key], self.args.sampling_rate)
            else:
                # 如果不使用键，直接将音频写入指定的输出路径
                if isinstance(self.result[key], list):  # 对于多说话者输出
                    for spk in range(self.args.num_spks):
                        sf.write(output_path.replace('.wav', f'_s{spk+1}.wav'),
                                 self.result[key][spk], self.args.sampling_rate)
                else:
                    sf.write(output_path, self.result[key], self.args.sampling_rate)

# 特定子任务的模型类

class CLS_FRCRN_SE_16K(SpeechModel):
    """
    SpeechModel 的子类，使用 FRCRN 架构实现 16 kHz 语音增强模型。
    
    参数:
        args (Namespace): 包含模型配置和路径的参数解析对象。
    """

    def __init__(self, args):
        # 初始化父类 SpeechModel
        super(CLS_FRCRN_SE_16K, self).__init__(args)
        
        # 导入 FRCRN 语音增强模型用于 16 kHz
        from models.frcrn_se.frcrn import FRCRN_SE_16K
        
        # 初始化模型
        self.model = FRCRN_SE_16K(args).model
        self.name = 'FRCRN_SE_16K'
        
        # 加载预训练模型检查点
        self.load_model()
        
        # 将模型移动到适当的设备（GPU/CPU）
        if args.use_cuda == 1:
            self.model.to(self.device)
        
        # 将模型设置为评估模式（不计算梯度）
        self.model.eval()

class CLS_MossFormer2_SE_48K(SpeechModel):
    """
    SpeechModel 的子类，使用 MossFormer2 架构实现 48 kHz 语音增强模型。
    
    参数:
        args (Namespace): 包含模型配置和路径的参数解析对象。
    """

    def __init__(self, args):
        # 初始化父类 SpeechModel
        super(CLS_MossFormer2_SE_48K, self).__init__(args)
        
        # 导入 MossFormer2 语音增强模型用于 48 kHz
        from models.mossformer2_se.mossformer2_se_wrapper import MossFormer2_SE_48K
        
        # 初始化模型
        self.model = MossFormer2_SE_48K(args).model
        self.name = 'MossFormer2_SE_48K'
        
        # 加载预训练模型检查点
        self.load_model()
        
        # 将模型移动到适当的设备（GPU/CPU）
        if args.use_cuda == 1:
            self.model.to(self.device)
        
        # 将模型设置为评估模式（不计算梯度）
        self.model.eval()

class CLS_AV_MossFormer2_TSE_16K(SpeechModel):
    """
    SpeechModel 的子类，使用 AV_MossFormer2 架构实现 16 kHz 目标说话人提取。
    该模型利用音频和视觉线索进行说话人提取。
    
    参数:
        args (Namespace): 包含模型配置和路径的参数解析对象。
    """

    def __init__(self, args):
        # 初始化父类 SpeechModel
        super(CLS_AV_MossFormer2_TSE_16K, self).__init__(args)
        
        # 导入 AV_MossFormer2 模型用于 16 kHz 目标语音增强
        from models.av_mossformer2_tse.av_mossformer2 import AV_MossFormer2_TSE_16K
        
        # 初始化模型
        self.model = AV_MossFormer2_TSE_16K(args).model
        self.name = 'AV_MossFormer2_TSE_16K'
        
        # 加载预训练模型检查点
        self.load_model()
        
        # 将模型移动到适当的设备（GPU/CPU）
        if args.use_cuda == 1:
            self.model.to(self.device)
        
        # 将模型设置为评估模式（不计算梯度）
        self.model.eval()