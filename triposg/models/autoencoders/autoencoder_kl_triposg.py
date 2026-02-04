from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm, LayerNorm
from diffusers.utils import logging
from diffusers.utils.accelerate_utils import apply_forward_hook
from einops import repeat
from torch_cluster import fps
from tqdm import tqdm

from ..attention_processor import FusedTripoSGAttnProcessor2_0, TripoSGAttnProcessor2_0, FlashTripoSGAttnProcessor2_0
from ..attention_processor import CustomAttnProcessor2_0
from ..embeddings import FrequencyPositionalEmbedding
from ..transformers.triposg_transformer import DiTBlock
from .vae import DiagonalGaussianDistribution

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

DEBUG_PRINT = False  # Set to True to enable debug prints
print("🐛🐛 DEBUG PRINT (autoencoder_kl_triposg.py)") if DEBUG_PRINT else None


def otsu_threshold_1d(data):
    """
    Otsu's method for 1D data to find optimal threshold
    
    Args:
        data: 1D numpy array of values
        
    Returns:
        optimal_threshold: the threshold value that maximizes inter-class variance
    """
    data = data.flatten()
    
    # Create histogram
    hist, bin_edges = np.histogram(data, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Normalize histogram
    hist = hist.astype(np.float32)
    hist = hist / np.sum(hist)
    
    # Compute cumulative sums
    cumsum = np.cumsum(hist)
    cumsum_mean = np.cumsum(hist * bin_centers)
    
    # Avoid division by zero
    cumsum = np.maximum(cumsum, 1e-10)
    
    # Compute total mean
    global_mean = np.sum(hist * bin_centers)
    
    # Compute between-class variance for each possible threshold
    between_class_variance = np.zeros_like(cumsum)
    
    for i in range(len(cumsum)):
        if cumsum[i] > 0 and cumsum[i] < 1:
            # Mean of class 1 (below threshold)
            mean1 = cumsum_mean[i] / cumsum[i]
            
            # Mean of class 2 (above threshold)  
            w2 = 1 - cumsum[i]
            if w2 > 0:
                mean2 = (global_mean - cumsum_mean[i]) / w2
                
                # Between-class variance
                between_class_variance[i] = cumsum[i] * w2 * (mean1 - mean2) ** 2
    
    # Find threshold that maximizes between-class variance
    optimal_idx = np.argmax(between_class_variance)
    optimal_threshold = bin_centers[optimal_idx]
    
    return optimal_threshold


class TripoSGEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        dim: int = 512,
        num_attention_heads: int = 8,
        num_layers: int = 8,
    ):
        super().__init__()

        self.proj_in = nn.Linear(in_channels, dim, bias=True)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    num_attention_heads=num_attention_heads,
                    use_self_attention=False,
                    use_cross_attention=True,
                    cross_attention_dim=dim,
                    cross_attention_norm_type="layer_norm",
                    activation_fn="gelu",
                    norm_type="fp32_layer_norm",
                    norm_eps=1e-5,
                    qk_norm=False,
                    qkv_bias=False,
                )  # cross attention
            ]
            + [
                DiTBlock(
                    dim=dim,
                    num_attention_heads=num_attention_heads,
                    use_self_attention=True,
                    self_attention_norm_type="fp32_layer_norm",
                    use_cross_attention=False,
                    activation_fn="gelu",
                    norm_type="fp32_layer_norm",
                    norm_eps=1e-5,
                    qk_norm=False,
                    qkv_bias=False,
                )
                for _ in range(num_layers)  # self attention
            ]
        )

        self.norm_out = LayerNorm(dim)

    def forward(self, sample_1: torch.Tensor, sample_2: torch.Tensor):
        hidden_states = self.proj_in(sample_1)
        encoder_hidden_states = self.proj_in(sample_2)

        for layer, block in enumerate(self.blocks):
            if layer == 0:
                hidden_states = block(
                    hidden_states, encoder_hidden_states=encoder_hidden_states
                )
            else:
                hidden_states = block(hidden_states)

        hidden_states = self.norm_out(hidden_states)

        return hidden_states


class dec_cross_forward_override:
    def __init__(self, vis_token=0):
        self.store_information = {"score_map": None, "attn_map": None}
        self.attn2_processor = CustomAttnProcessor2_0()
        self.grid_matrix= None
        self.xyz_queries = None
        self.vis_token = vis_token  # 用於可視化的 token，默認為 0
        self.vis_xyz_val=[]
        self.call_count=0
        # self.rescale_xyz_metrics = torch.linspace(-1.0049, 1.0049, int(256), dtype=torch.float16)

    def register(self, block):
        # overwrite block.attn2.preocessor
        block.attn2.set_processor(self.attn2_processor)
        # 使用閉包捕獲 self（override 實例)
        def custom_forward(
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_hidden_states_2: Optional[torch.Tensor] = None,
            temb: Optional[torch.Tensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None,
            skip: Optional[torch.Tensor] = None,
            attention_kwargs= None,
            ) -> torch.Tensor:
            self.call_count+=1
            assert isinstance(self.grid_matrix, torch.Tensor), "grid_matrix must be set before calling custom_forward"
            assert isinstance(self.xyz_queries, torch.Tensor), "xyz_queries must be set before calling custom_forward"
            if self.store_information.get("score_map") is None:
                self.store_information["score_map"] = self.grid_matrix.clone()
                print("🐛 in custom_forward creating 3d attn map:", self.store_information.get("score_map").shape)
            if self.store_information.get("attn_map") is None:
                self.store_information["attn_map"] = self.grid_matrix.clone()

            # 這裡可以同時訪問 self（override 實例）和 block
            #self.store_information["hidden_states"] = hidden_states
        
            # Prepare attention kwargs
            attention_kwargs = attention_kwargs.copy() if attention_kwargs is not None else {}
            cross_attention_scale = attention_kwargs.pop("cross_attention_scale", 1.0)
            cross_attention_2_scale = attention_kwargs.pop("cross_attention_2_scale", 1.0)

            # Notice that normalization is always applied before the real computation in the following blocks.
            # 0. Long Skip Connection
            if block.skip_linear is not None:
                cat = torch.cat(
                    (
                        [skip, hidden_states]
                        if block.skip_concat_front
                        else [hidden_states, skip]
                    ),
                    dim=-1,
                )
                if block.skip_norm_last:
                    # don't do this
                    hidden_states = block.skip_linear(cat)
                    hidden_states = block.skip_norm(hidden_states)
                else:
                    cat = block.skip_norm(cat)
                    hidden_states = block.skip_linear(cat)

            # 1. Self-Attention
            if block.use_self_attention:
                norm_hidden_states = block.norm1(hidden_states)
                attn_output = block.attn1(
                    norm_hidden_states,
                    image_rotary_emb=image_rotary_emb,
                    **attention_kwargs,
                )
                hidden_states = hidden_states + attn_output

            # 2. Cross-Attention
            if block.use_cross_attention:
                if block.use_cross_attention_2:
                    hidden_states = (
                        hidden_states
                        + block.attn2(
                            block.norm2(hidden_states),
                            encoder_hidden_states=encoder_hidden_states,
                            image_rotary_emb=image_rotary_emb,
                            **attention_kwargs,
                        ) * cross_attention_scale
                        + block.attn2_2(
                            block.norm2_2(hidden_states),
                            encoder_hidden_states=encoder_hidden_states_2,
                            image_rotary_emb=image_rotary_emb,
                            **attention_kwargs,
                        ) * cross_attention_2_scale
                    )
                else:
                    # This attn2 is overide by custom_attention processor
                    attn_output= block.attn2(
                        block.norm2(hidden_states),
                        encoder_hidden_states=encoder_hidden_states,
                        image_rotary_emb=image_rotary_emb,
                        **attention_kwargs,
                    ) * cross_attention_scale
                    hidden_states = hidden_states + attn_output

                    if DEBUG_PRINT:
                        attn_probs = self.attn2_processor.attn_probs
                        attn_probs_val=attn_probs[0].mean(dim=0)
                        # print("🐛 in custom_forward xyz_queries output:", self.xyz_queries.shape)
                        self.vis_xyz_val.append(attn_probs_val[:,self.vis_token].cpu())
                        # for xyz_index, xyz in enumerate(self.xyz_queries[0]):
                        #     print("xyz", xyz)
                        #     self.store_information["attn_map"][xyz[0], xyz[1], xyz[2]] = attn_probs_val[xyz_index, self.vis_token]

            # FFN Layer ### TODO: switch norm2 and norm3 in the state dict
            mlp_inputs = block.norm3(hidden_states)
            hidden_states = hidden_states + block.ff(mlp_inputs)

            return hidden_states
            
        # 直接賦值，不需要 __get__
        block.forward = custom_forward

class TripoSGDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,  # 輸入查詢點的維度（通常是 xyz 座標）
        out_channels: int = 1,  # 輸出維度（通常是 SDF 值）
        dim: int = 512,  # 模型的隱藏維度
        num_attention_heads: int = 8,  # 注意力頭數
        num_layers: int = 16,  # 自注意力層數
        grad_type: str = "analytical",  # 梯度計算方式："numerical" 或 "analytical"
        grad_interval: float = 0.001,  # 數值梯度的間隔
        debug_print: bool = DEBUG_PRINT,  # 是否打印調試信息
    ):
        super().__init__()
        self.debug_print = debug_print
        if grad_type not in ["numerical", "analytical"]:
            raise ValueError(f"grad_type must be one of ['numerical', 'analytical']")
        self.grad_type = grad_type
        
        self.grad_interval = grad_interval

        # 構建 Transformer 塊：前 num_layers 個是自注意力，最後一個是交叉注意力
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    num_attention_heads=num_attention_heads,
                    use_self_attention=True,  # 使用自注意力進行特徵提取
                    self_attention_norm_type="fp32_layer_norm",
                    use_cross_attention=False,
                    activation_fn="gelu",
                    norm_type="fp32_layer_norm",
                    norm_eps=1e-5,
                    qk_norm=False,
                    qkv_bias=False,
                )
                for _ in range(num_layers)  # 多層自注意力用於處理 latent features
            ]
            + [
                DiTBlock(
                    dim=dim,
                    num_attention_heads=num_attention_heads,
                    use_self_attention=False,
                    use_cross_attention=True,  # 最後一層使用交叉注意力，為了可以跟grid query compute geometry
                    cross_attention_dim=dim,
                    cross_attention_norm_type="layer_norm",
                    activation_fn="gelu",
                    norm_type="fp32_layer_norm",
                    norm_eps=1e-5,
                    qk_norm=False,
                    qkv_bias=False,
                )  # 交叉注意力用於查詢特定位置的幾何資訊
            ]
        )

        if self.debug_print:
            print(f"🐛‼️ Debug mode (TripoSGDecoder): Using custom forward for last block")
            self.override = dec_cross_forward_override()
            self.override.register(self.blocks[-1])

        # 將查詢點投影到模型維度
        self.proj_query = nn.Linear(in_channels, dim, bias=True)

        # 輸出層
        self.norm_out = LayerNorm(dim)
        self.proj_out = nn.Linear(dim, out_channels, bias=True)


    def set_topk(self, topk):
        """設定最後一層交叉注意力的 top-k 參數以節省記憶體"""
        self.blocks[-1].set_topk(topk)

    def set_flash_processor(self, processor):
        """設定 Flash Attention 處理器以加速計算"""
        self.blocks[-1].set_flash_processor(processor)

    def query_geometry(
        self,
        model_fn: callable,
        queries: torch.Tensor,  # 查詢點座標 (B, N, 3)
        sample: torch.Tensor,   # 編碼後的潛在特徵 (B, M, dim)
        grad: bool = False,     # 是否計算梯度
    ):
        """
        查詢特定位置的幾何資訊（如 SDF 值）
        
        Args:
            model_fn: 模型函數，接受查詢點和潛在特徵，返回預測值
            queries: 查詢點座標
            sample: 編碼後的潛在特徵
            grad: 是否計算梯度（用於表面法向量計算）
        
        Returns:
            logits: 預測的幾何值
            grad_value: 梯度值（如果 grad=True）
        """
        # 獲取基本的幾何預測值
        logits = model_fn(queries, sample)
        
        if grad:
            # 計算梯度用於表面法向量或其他幾何資訊
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                if self.grad_type == "numerical":
                    # 數值梯度：使用有限差分法
                    interval = self.grad_interval
                    grad_value = []
                    for offset in [
                        (interval, 0, 0),  # x 方向
                        (0, interval, 0),  # y 方向
                        (0, 0, interval),  # z 方向
                    ]:
                        offset_tensor = torch.tensor(offset, device=queries.device)[
                            None, :
                        ]
                        # 計算正向和負向的函數值
                        res_p = model_fn(queries + offset_tensor, sample)[..., 0]
                        res_n = model_fn(queries - offset_tensor, sample)[..., 0]
                        # 計算梯度
                        grad_value.append((res_p - res_n) / (2 * interval))
                    grad_value = torch.stack(grad_value, dim=-1)
                else:
                    # 解析梯度：使用自動微分
                    queries_d = torch.clone(queries)
                    queries_d.requires_grad = True
                    with torch.enable_grad():
                        res_d = model_fn(queries_d, sample)
                        grad_value = torch.autograd.grad(
                            res_d,
                            [queries_d],
                            grad_outputs=torch.ones_like(res_d),
                            create_graph=self.training,  # 訓練時保留計算圖
                        )[0]
        else:
            grad_value = None

        return logits, grad_value

    def forward(
        self,
        sample: torch.Tensor,     # 潛在特徵 (B, M, dim)
        queries: torch.Tensor,    # 查詢點座標 (B, N, 3)
        kv_cache: Optional[torch.Tensor] = None,  # 快取的鍵值對
    ):
        """
        Vec2Set Decoder 的前向傳播
        
        工作流程：
        1. 使用自注意力層處理潛在特徵，提取全局幾何資訊
        2. 使用交叉注意力查詢特定位置的幾何值
        3. 可選地計算梯度用於表面法向量
        
        Args:
            sample: 來自編碼器的潛在特徵
            queries: 要查詢的 3D 座標點
            kv_cache: 快取的特徵（避免重複計算）
        
        Returns:
            logits: 查詢點的幾何預測值
            kv_cache: 處理後的特徵快取
        """
        if kv_cache is None:
            
            # 第一次調用：通過自注意力層處理潛在特徵
            # 因為我們沒辦法一次處裡所有的grid points，所以需要分批處理
            # 但是潛在特徵都是一樣的，所以我們只需要forward 潛在特徵 with self-attention
            hidden_states = sample
            for _, block in enumerate(self.blocks[:-1]):
                # 使用自注意力提取和整合全局幾何特徵
                hidden_states = block(hidden_states)
            kv_cache = hidden_states  # 快取處理後的特徵

    

        # 定義查詢函數：使用交叉注意力查詢特定位置的幾何資訊
        def query_fn(q, kv):
            # 將查詢點投影到模型維度
            q = self.proj_query(q)
            # 使用交叉注意力：查詢點作為 query，潛在特徵作為 key 和 value
            l = self.blocks[-1](q, encoder_hidden_states=kv)
            # 輸出幾何預測值
            return self.proj_out(self.norm_out(l))
        
        if self.debug_print:
            def query_fn(q,kv):
                # 將查詢點投影到模型維度
                q = self.proj_query(q)
                # 使用交叉注意力：查詢點作為 query，潛在特徵作為 key 和 value
                l = self.blocks[-1](q, encoder_hidden_states=kv)
                # 輸出幾何預測值
                return self.proj_out(self.norm_out(l)), None

        # 執行幾何查詢
        logits, grad = self.query_geometry(
            query_fn, queries, kv_cache, grad=self.training
        )
        
        # 將 logits 取負值（可能是因為使用 SDF 的慣例）
        logits = logits * -1 if not isinstance(logits, Tuple) else logits[0] * -1

        return logits, kv_cache


class TripoSGVAEModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,  # NOTE xyz instead of feature dim
        latent_channels: int = 64,
        num_attention_heads: int = 8,
        width_encoder: int = 512,
        width_decoder: int = 1024,
        num_layers_encoder: int = 8,
        num_layers_decoder: int = 16,
        embedding_type: str = "frequency",
        embed_frequency: int = 8,
        embed_include_pi: bool = False,
    ):
        super().__init__()

        self.out_channels = 1

        if embedding_type == "frequency":
            self.embedder = FrequencyPositionalEmbedding(
                num_freqs=embed_frequency,
                logspace=True,
                input_dim=in_channels,
                include_pi=embed_include_pi,
            )
        else:
            raise NotImplementedError(
                f"Embedding type {embedding_type} is not supported."
            )

        self.encoder = TripoSGEncoder(
            in_channels=in_channels + self.embedder.out_dim,
            dim=width_encoder,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers_encoder,
        )
        self.decoder = TripoSGDecoder(
            in_channels=self.embedder.out_dim,
            out_channels=self.out_channels,
            dim=width_decoder,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers_decoder,
        )

        self.quant = nn.Linear(width_encoder, latent_channels * 2, bias=True)
        self.post_quant = nn.Linear(latent_channels, width_decoder, bias=True)

        self.use_slicing = False
        self.slicing_length = 1

    def set_flash_decoder(self):
        self.decoder.set_flash_processor(FlashTripoSGAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedTripoSGAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError(
                    "`fuse_qkv_projections()` is not supported for models having added KV projections."
                )

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedTripoSGAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(
            name: str,
            module: torch.nn.Module,
            processors: Dict[str, AttentionProcessor],
        ):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]
    ):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        self.set_attn_processor(TripoSGAttnProcessor2_0())

    def enable_slicing(self, slicing_length: int = 1) -> None:
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True
        self.slicing_length = slicing_length

    def disable_slicing(self) -> None:
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    def _sample_features(
        self, x: torch.Tensor, num_tokens: int = 2048, seed: Optional[int] = None
    ):
        """
        Sample points from features of the input point cloud.

        Args:
            x (torch.Tensor): The input point cloud. shape: (B, N, C)
            num_tokens (int, optional): The number of points to sample. Defaults to 2048.
            seed (Optional[int], optional): The random seed. Defaults to None.
        """
        rng = np.random.default_rng(seed)
        indices = rng.choice(
            x.shape[1], num_tokens * 4, replace=num_tokens * 4 > x.shape[1]
        )
        selected_points = x[:, indices]

        batch_size, num_points, num_channels = selected_points.shape
        flattened_points = selected_points.view(batch_size * num_points, num_channels)
        batch_indices = (
            torch.arange(batch_size).to(x.device).repeat_interleave(num_points)
        )

        # fps sampling with fallback for CUDA compatibility issues
        sampling_ratio = 1.0 / 4
        
        sampled_indices = fps(
            flattened_points[:, :3],
            batch_indices,
            ratio=sampling_ratio,
            random_start=self.training,
        )
        sampled_points = flattened_points[sampled_indices].view(
            batch_size, -1, num_channels
        )

        return sampled_points

    def _encode(
        self, x: torch.Tensor, num_tokens: int = 2048, seed: Optional[int] = None
    ):
        position_channels = self.config.in_channels
        positions, features = x[..., :position_channels], x[..., position_channels:]
        x_kv = torch.cat([self.embedder(positions), features], dim=-1)
        sampled_x = self._sample_features(x, num_tokens, seed)
        positions, features = (
            sampled_x[..., :position_channels],
            sampled_x[..., position_channels:],
        )

        x_q = torch.cat([self.embedder(positions), features], dim=-1)

        x = self.encoder(x_q, x_kv)


        x = self.quant(x)

        return x

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, return_dict: bool = True, **kwargs
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        """
        Encode a batch of point features into latents.
        """
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [
                self._encode(x_slice, **kwargs)
                for x_slice in x.split(self.slicing_length)
            ]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x, **kwargs)

        posterior = DiagonalGaussianDistribution(h, feature_dim=-1)

        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(
        self,
        z: torch.Tensor,
        sampled_points: torch.Tensor,
        num_chunks: int = 50000,
        to_cpu: bool = False,
        return_dict: bool = True,
        debug_print=DEBUG_PRINT,
    ) -> Union[DecoderOutput, torch.Tensor]:
        
        xyz_samples = sampled_points
        if debug_print:
            print("🐛 Debug mode (_decode)")if debug_print else None
            print("xyz_samples.shape:", xyz_samples.shape) if debug_print else None
            print("z.shape before post_quant:", z.shape) if debug_print else None
            grid_dim = round(xyz_samples.shape[1]**(1/3))
            #print("xyz_samples.shape:", xyz_samples.shape) if debug_print else None
            self.decoder.override.grid_matrix=torch.zeros(
                grid_dim, grid_dim, grid_dim, device=xyz_samples.device
            )
            print("grid_dim:", grid_dim) if debug_print else None
        z = self.post_quant(z)        

        num_points = xyz_samples.shape[1]
        kv_cache = None
        dec = []

        for i in range(0, num_points, num_chunks):
            # provide current xyz information to the decoder
            # self.decoder.override.xyz_queries=xyz_samples[:, i : i + num_chunks, :].to(z.device, dtype=z.dtype)
            queries = xyz_samples[:, i : i + num_chunks, :].to(z.device, dtype=z.dtype)
            queries = self.embedder(queries)
            z_, kv_cache = self.decoder(z, queries, kv_cache)
            dec.append(z_ if not to_cpu else z_.cpu())

        z = torch.cat(dec, dim=1)
        if debug_print:
            print(f"concatenating  overide attn score")
            out_attn_probs = torch.cat(
                [d for d in self.decoder.override.vis_xyz_val], dim=0
            )
            print(f"out_attn_probs: {out_attn_probs.shape}")
            
            # Apply Otsu's thresholding to find optimal cutting point
            otsu_threshold = otsu_threshold_1d(out_attn_probs.cpu().numpy())
            print(f"🔍 Otsu optimal threshold: {otsu_threshold:.6f}")
            
            # Apply threshold to create binary mask
            binary_mask = out_attn_probs > otsu_threshold
            num_positive = binary_mask.sum().item()
            total_points = out_attn_probs.numel()
            print(f"📊 Points above threshold: {num_positive}/{total_points} ({100*num_positive/total_points:.2f}%)")
            
            grid_dim = round(xyz_samples.shape[1]**(1/3))
            out_attn_probs = out_attn_probs.to(torch.float16).view(grid_dim, grid_dim, grid_dim)
            print("🐛👽Visualizing xyz attention map with shape:", out_attn_probs.shape)
            print("max, min attn probs:", out_attn_probs.max(), out_attn_probs.min())
            save_grid_logits_as_pointcloud(
                grid_logits=out_attn_probs,
                output_path=f"./output/attn_cloud/demo_{self.decoder.override.call_count}",
                format_type="ply",
                threshold=otsu_threshold,  # Use Otsu threshold instead of fixed value
                downsample_factor=2,  # 下採樣減少點數量
                colormap="coolwarm" 
            )
            
            # Also save binary mask as point cloud for visualization
            binary_mask_3d = binary_mask.to(torch.float16).view(grid_dim, grid_dim, grid_dim)
            save_grid_logits_as_pointcloud(
                grid_logits=binary_mask_3d,
                output_path=f"./output/attn_cloud/demo_binary_{self.decoder.override.call_count}",
                format_type="ply",
                threshold=0.5,  # Binary threshold
                downsample_factor=2,
                colormap="RdBu_r"  # Red-Blue colormap for binary visualization
            )
            # Release override attn score
            print("realease override attn score")
            self.decoder.override.vis_xyz_val = []

        if not return_dict:
            return (z,)

        return DecoderOutput(sample=z)

    @apply_forward_hook
    def decode(
        self,
        z: torch.Tensor,
        sampled_points: torch.Tensor,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[DecoderOutput, torch.Tensor]:
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [
                self._decode(z_slice, p_slice, **kwargs).sample
                for z_slice, p_slice in zip(
                    z.split(self.slicing_length),
                    sampled_points.split(self.slicing_length),
                )
            ]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z, sampled_points, **kwargs).sample

        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def forward(self, x: torch.Tensor):
        pass



def save_grid_logits_as_pointcloud(grid_logits: torch.Tensor, 
                                   output_path: str = "grid_logits_pointcloud",
                                   format_type: str = "ply",
                                   threshold: float = -5.0,
                                   downsample_factor: int = 1,
                                   colormap: str = "coolwarm"):
    """
    將3D grid logits轉換為點雲格式並儲存
    
    適合的3D點雲格式：
    1. PLY (Polygon File Format) - 最常用，支援顏色
    2. PCD (Point Cloud Data) - PCL library格式
    3. XYZ - 簡單文字格式
    4. OFF - Object File Format
    
    Args:
        grid_logits: shape (H, W, D) 的3D tensor
        output_path: 輸出檔案路徑（不含副檔名）
        format_type: 格式類型 ("ply", "xyz", "pcd")
        threshold: 只保存大於此閾值的點（過濾背景）
        downsample_factor: 下採樣因子，減少點的數量
        colormap: 色彩映射類型 ("viridis", "plasma", "inferno", "magma", "hot", "coolwarm", "RdYlBu_r")
    """
    
    if isinstance(grid_logits, torch.Tensor):
        grid_logits = grid_logits.cpu().numpy()
    
    H, W, D = grid_logits.shape
    print(f"Original grid shape: {grid_logits.shape}")
    print(f"Value range: {grid_logits.min():.3f} to {grid_logits.max():.3f}")
    
    # 生成3D座標網格
    x_coords, y_coords, z_coords = np.meshgrid(
        np.arange(0, H, downsample_factor),
        np.arange(0, W, downsample_factor), 
        np.arange(0, D, downsample_factor),
        indexing='ij'
    )
    
    # 下採樣grid_logits
    downsampled_logits = grid_logits[::downsample_factor, ::downsample_factor, ::downsample_factor]
    
    # 扁平化所有數據
    points = np.stack([x_coords.flatten(), y_coords.flatten(), z_coords.flatten()], axis=1)
    values = downsampled_logits.flatten()
    
    # 根據閾值過濾點
    mask = values > threshold
    filtered_points = points[mask]
    filtered_values = values[mask]
    
    print(f"Filtered points: {len(filtered_points)} / {len(points)}")
    
    # 將數值映射到顏色（使用matplotlib heatmap顏色映射）
    # 正規化到0-1範圍
    min_val, max_val = filtered_values.min(), filtered_values.max()
    normalized_values = (filtered_values - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(filtered_values)
    
    # 使用matplotlib viridis色彩映射 (深藍->綠->黃->紅)
    # 也可以選擇其他色彩映射: 'plasma', 'inferno', 'magma', 'cividis', 'hot', 'coolwarm'
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    # 選擇色彩映射 - 提供多種heatmap顏色選擇
    colormap_obj = cm.get_cmap(colormap)
    
    # 將正規化值映射到RGB顏色
    colors_rgba = colormap_obj(normalized_values)
    colors = (colors_rgba[:, :3] * 255).astype(np.uint8)  # 轉換為0-255範圍的RGB
    
    if format_type.lower() == "ply":
        output_file = f"{output_path}.ply"
        save_ply_pointcloud(filtered_points, colors, filtered_values, output_file)
    elif format_type.lower() == "xyz":
        output_file = f"{output_path}.xyz"
        save_xyz_pointcloud(filtered_points, colors, filtered_values, output_file)
    elif format_type.lower() == "pcd":
        output_file = f"{output_path}.pcd"
        save_pcd_pointcloud(filtered_points, colors, filtered_values, output_file)
    else:
        raise ValueError(f"Unsupported format: {format_type}")
    
    print(f"Point cloud saved to: {output_file}")
    return output_file

def save_ply_pointcloud(points, colors, values, output_path):
    """儲存為PLY格式（最推薦，支援多數3D軟體）"""
    with open(output_path, 'w') as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property float value\n")
        f.write("end_header\n")
        
        # 點數據
        for i in range(len(points)):
            f.write(f"{points[i,0]:.3f} {points[i,1]:.3f} {points[i,2]:.3f} "
                   f"{colors[i,0]} {colors[i,1]} {colors[i,2]} {values[i]:.6f}\n")

def save_xyz_pointcloud(points, colors, values, output_path):
    """儲存為XYZ格式（簡單文字格式）"""
    with open(output_path, 'w') as f:
        for i in range(len(points)):
            f.write(f"{points[i,0]:.3f} {points[i,1]:.3f} {points[i,2]:.3f} "
                   f"{colors[i,0]} {colors[i,1]} {colors[i,2]} {values[i]:.6f}\n")

def save_pcd_pointcloud(points, colors, values, output_path):
    """儲存為PCD格式（PCL library格式）"""
    with open(output_path, 'w') as f:
        # PCD header
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z rgb value\n")
        f.write("SIZE 4 4 4 4 4\n")
        f.write("TYPE F F F U F\n")
        f.write("COUNT 1 1 1 1 1\n")
        f.write(f"WIDTH {len(points)}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(points)}\n")
        f.write("DATA ascii\n")
        
        # 點數據
        for i in range(len(points)):
            # 將RGB打包成單一整數
            rgb = (int(colors[i,0]) << 16) | (int(colors[i,1]) << 8) | int(colors[i,2])
            f.write(f"{points[i,0]:.3f} {points[i,1]:.3f} {points[i,2]:.3f} {rgb} {values[i]:.6f}\n")