"""
Inference Engine Package

Advanced inference optimizations and acceleration techniques.
"""

from .speculative_decoding import (
    SpeculativeDecoder,
    SpeculativeConfig,
    create_speculative_decoder,
    integrate_speculative_decoding
)

from .tensorrt_triton import (
    TensorRTOptimizer,
    TritonModelDeployer,
    InferenceAccelerator,
    TensorRTConfig,
    TritonConfig,
    create_tensorrt_optimizer,
    create_triton_deployer,
    integrate_with_existing_models
)

from .dynamic_batching import (
    DynamicBatcher,
    AsyncStreamingServer,
    BatchConfig,
    BatchRequest,
    create_dynamic_batcher,
    create_streaming_server,
    integrate_dynamic_batching
)

from .quantization import (
    ModelQuantizer,
    AdvancedQuantizer,
    QuantizationConfig,
    create_quantizer,
    create_advanced_quantizer,
    integrate_quantization
)

__all__ = [
    # Speculative Decoding
    'SpeculativeDecoder',
    'SpeculativeConfig', 
    'create_speculative_decoder',
    'integrate_speculative_decoding',
    
    # TensorRT/Triton
    'TensorRTOptimizer',
    'TritonModelDeployer',
    'InferenceAccelerator',
    'TensorRTConfig',
    'TritonConfig',
    'create_tensorrt_optimizer',
    'create_triton_deployer',
    'integrate_with_existing_models',
    
    # Dynamic Batching
    'DynamicBatcher',
    'AsyncStreamingServer',
    'BatchConfig',
    'BatchRequest',
    'create_dynamic_batcher',
    'create_streaming_server',
    'integrate_dynamic_batching',
    
    # Quantization
    'ModelQuantizer',
    'AdvancedQuantizer',
    'QuantizationConfig',
    'create_quantizer',
    'create_advanced_quantizer',
    'integrate_quantization'
]
