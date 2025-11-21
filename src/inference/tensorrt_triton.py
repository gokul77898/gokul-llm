"""
TensorRT/Triton Inference Server Integration Hooks

Provides backend-ready hooks for TensorRT optimization and Triton Inference Server deployment.
Includes fallback mechanisms for Mac/CPU environments.
"""

import torch
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)


@dataclass
class TensorRTConfig:
    """Configuration for TensorRT optimization"""
    precision: str = "fp16"  # fp32, fp16, int8
    max_batch_size: int = 8
    max_sequence_length: int = 2048
    workspace_size: int = 1 << 30  # 1GB
    enable_dynamic_shapes: bool = True
    optimization_level: int = 3  # 0-5, higher = more aggressive
    cache_dir: str = "tensorrt_cache"


@dataclass
class TritonConfig:
    """Configuration for Triton Inference Server"""
    model_name: str = "mark_ai_model"
    model_version: str = "1"
    max_batch_size: int = 8
    dynamic_batching: bool = True
    instance_count: int = 1
    backend: str = "pytorch"  # pytorch, tensorrt, onnx
    model_repository: str = "triton_models"


class TensorRTOptimizer:
    """
    TensorRT Optimization Engine
    
    Provides hooks for TensorRT optimization with fallback for non-CUDA environments.
    """
    
    def __init__(self, config: TensorRTConfig):
        self.config = config
        self.is_available = self._check_tensorrt_availability()
        self.optimized_engines = {}
        
        if self.is_available:
            logger.info("✅ TensorRT available - optimization enabled")
        else:
            logger.info("⚠️  TensorRT not available - using fallback mode")
    
    def _check_tensorrt_availability(self) -> bool:
        """Check if TensorRT is available"""
        try:
            import tensorrt as trt
            import torch_tensorrt
            
            # Check CUDA availability
            if not torch.cuda.is_available():
                logger.info("CUDA not available - TensorRT disabled")
                return False
            
            # Check TensorRT version
            trt_version = trt.__version__
            logger.info(f"TensorRT version: {trt_version}")
            return True
            
        except ImportError:
            logger.info("TensorRT not installed - install with: pip install torch-tensorrt tensorrt")
            return False
        except Exception as e:
            logger.warning(f"TensorRT check failed: {e}")
            return False
    
    def optimize_model(
        self, 
        model: torch.nn.Module,
        example_inputs: torch.Tensor,
        model_name: str = "optimized_model"
    ) -> Union[torch.nn.Module, None]:
        """
        Optimize model with TensorRT
        
        Args:
            model: PyTorch model to optimize
            example_inputs: Example input tensor for optimization
            model_name: Name for the optimized model
            
        Returns:
            Optimized model or None if optimization fails
        """
        if not self.is_available:
            logger.info("TensorRT not available - returning original model")
            return model
        
        try:
            import torch_tensorrt
            
            logger.info(f"🚀 Optimizing model '{model_name}' with TensorRT")
            logger.info(f"   Precision: {self.config.precision}")
            logger.info(f"   Max batch size: {self.config.max_batch_size}")
            
            # Prepare model
            model.eval()
            model = model.cuda()
            
            # Configure TensorRT compilation
            compile_spec = {
                "inputs": [
                    torch_tensorrt.Input(
                        min_shape=example_inputs.shape,
                        opt_shape=example_inputs.shape,
                        max_shape=tuple(
                            self.config.max_batch_size if i == 0 else s 
                            for i, s in enumerate(example_inputs.shape)
                        )
                    )
                ],
                "enabled_precisions": self._get_precision_set(),
                "workspace_size": self.config.workspace_size,
                "max_aux_streams": 4,
                "refit": False,
                "debug": False,
                "device": {
                    "device_type": torch_tensorrt.DeviceType.GPU,
                    "gpu_id": 0
                }
            }
            
            # Compile model
            start_time = time.time()
            optimized_model = torch_tensorrt.compile(model, **compile_spec)
            optimization_time = time.time() - start_time
            
            # Cache the optimized model
            self.optimized_engines[model_name] = optimized_model
            
            logger.info(f"✅ TensorRT optimization complete")
            logger.info(f"   Optimization time: {optimization_time:.2f}s")
            
            return optimized_model
            
        except Exception as e:
            logger.error(f"❌ TensorRT optimization failed: {e}")
            logger.info("   Falling back to original model")
            return model
    
    def _get_precision_set(self) -> set:
        """Get TensorRT precision set based on config"""
        try:
            import torch_tensorrt
            
            if self.config.precision == "fp32":
                return {torch.float32}
            elif self.config.precision == "fp16":
                return {torch.float16, torch.float32}
            elif self.config.precision == "int8":
                return {torch.int8, torch.float16, torch.float32}
            else:
                return {torch.float32}
        except:
            return {torch.float32}
    
    def save_engine(self, model_name: str, save_path: str) -> bool:
        """Save optimized TensorRT engine"""
        if model_name not in self.optimized_engines:
            logger.warning(f"Model '{model_name}' not found in optimized engines")
            return False
        
        try:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save TensorRT engine
            engine_path = save_path / f"{model_name}.engine"
            torch.jit.save(self.optimized_engines[model_name], str(engine_path))
            
            # Save metadata
            metadata = {
                "model_name": model_name,
                "precision": self.config.precision,
                "max_batch_size": self.config.max_batch_size,
                "max_sequence_length": self.config.max_sequence_length,
                "optimization_time": time.time()
            }
            
            metadata_path = save_path / f"{model_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ TensorRT engine saved: {engine_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save TensorRT engine: {e}")
            return False
    
    def benchmark_model(
        self, 
        model: torch.nn.Module, 
        example_inputs: torch.Tensor,
        num_runs: int = 100
    ) -> Dict[str, float]:
        """Benchmark model performance"""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available for benchmarking")
            return {"error": "CUDA not available"}
        
        try:
            model.eval()
            model = model.cuda()
            example_inputs = example_inputs.cuda()
            
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = model(example_inputs)
            
            torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_runs):
                with torch.no_grad():
                    _ = model(example_inputs)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_latency = total_time / num_runs * 1000  # ms
            throughput = num_runs / total_time  # inferences/sec
            
            return {
                "avg_latency_ms": avg_latency,
                "throughput_fps": throughput,
                "total_time_s": total_time,
                "num_runs": num_runs
            }
            
        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")
            return {"error": str(e)}


class TritonModelDeployer:
    """
    Triton Inference Server Model Deployer
    
    Prepares models for deployment on Triton Inference Server.
    """
    
    def __init__(self, config: TritonConfig):
        self.config = config
        self.model_repo_path = Path(config.model_repository)
        self.model_repo_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("📦 Triton Model Deployer initialized")
        logger.info(f"   Model repository: {self.model_repo_path}")
    
    def prepare_pytorch_model(
        self,
        model: torch.nn.Module,
        model_name: Optional[str] = None
    ) -> bool:
        """
        Prepare PyTorch model for Triton deployment
        
        Args:
            model: PyTorch model to deploy
            model_name: Name for the model (uses config default if None)
            
        Returns:
            True if preparation successful
        """
        model_name = model_name or self.config.model_name
        
        try:
            # Create model directory structure
            model_dir = self.model_repo_path / model_name
            version_dir = model_dir / self.config.model_version
            version_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_path = version_dir / "model.pt"
            torch.jit.save(torch.jit.script(model), str(model_path))
            
            # Create config.pbtxt
            config_content = self._generate_pytorch_config(model_name)
            config_path = model_dir / "config.pbtxt"
            
            with open(config_path, 'w') as f:
                f.write(config_content)
            
            logger.info(f"✅ PyTorch model prepared for Triton: {model_dir}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare PyTorch model: {e}")
            return False
    
    def prepare_tensorrt_model(
        self,
        engine_path: str,
        model_name: Optional[str] = None
    ) -> bool:
        """
        Prepare TensorRT engine for Triton deployment
        
        Args:
            engine_path: Path to TensorRT engine file
            model_name: Name for the model
            
        Returns:
            True if preparation successful
        """
        model_name = model_name or self.config.model_name
        
        try:
            import shutil
            
            # Create model directory structure
            model_dir = self.model_repo_path / f"{model_name}_tensorrt"
            version_dir = model_dir / self.config.model_version
            version_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy TensorRT engine
            engine_dest = version_dir / "model.plan"
            shutil.copy2(engine_path, engine_dest)
            
            # Create config.pbtxt
            config_content = self._generate_tensorrt_config(f"{model_name}_tensorrt")
            config_path = model_dir / "config.pbtxt"
            
            with open(config_path, 'w') as f:
                f.write(config_content)
            
            logger.info(f"✅ TensorRT model prepared for Triton: {model_dir}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare TensorRT model: {e}")
            return False
    
    def _generate_pytorch_config(self, model_name: str) -> str:
        """Generate Triton config for PyTorch model"""
        return f'''
name: "{model_name}"
backend: "pytorch"
max_batch_size: {self.config.max_batch_size}

input [
  {{
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }}
]

output [
  {{
    name: "logits"
    data_type: TYPE_FP32
    dims: [ -1, -1 ]
  }}
]

dynamic_batching {{
  max_queue_delay_microseconds: 100
}}

instance_group [
  {{
    count: {self.config.instance_count}
    kind: KIND_GPU
  }}
]
'''
    
    def _generate_tensorrt_config(self, model_name: str) -> str:
        """Generate Triton config for TensorRT model"""
        return f'''
name: "{model_name}"
backend: "tensorrt"
max_batch_size: {self.config.max_batch_size}

input [
  {{
    name: "input_ids"
    data_type: TYPE_INT32
    dims: [ -1 ]
  }}
]

output [
  {{
    name: "logits"
    data_type: TYPE_FP32
    dims: [ -1, -1 ]
  }}
]

dynamic_batching {{
  max_queue_delay_microseconds: 50
}}

optimization {{
  execution_accelerators {{
    gpu_execution_accelerator : [ {{
      name : "tensorrt"
      parameters {{ key: "precision_mode" value: "FP16" }}
      parameters {{ key: "max_workspace_size_bytes" value: "1073741824" }}
    }} ]
  }}
}}

instance_group [
  {{
    count: {self.config.instance_count}
    kind: KIND_GPU
  }}
]
'''
    
    def generate_client_code(self, model_name: str) -> str:
        """Generate sample client code for the deployed model"""
        return f'''
import tritonclient.http as httpclient
import numpy as np

# Create Triton client
client = httpclient.InferenceServerClient(url="localhost:8000")

# Prepare input
input_ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
inputs = [httpclient.InferInput("input_ids", input_ids.shape, "INT64")]
inputs[0].set_data_from_numpy(input_ids)

# Prepare output
outputs = [httpclient.InferRequestedOutput("logits")]

# Run inference
response = client.infer(model_name="{model_name}", inputs=inputs, outputs=outputs)
logits = response.as_numpy("logits")

print(f"Output shape: {{logits.shape}}")
print(f"Output: {{logits}}")
'''


class InferenceAccelerator:
    """
    Unified Inference Acceleration Manager
    
    Combines TensorRT optimization and Triton deployment capabilities.
    """
    
    def __init__(
        self,
        tensorrt_config: Optional[TensorRTConfig] = None,
        triton_config: Optional[TritonConfig] = None
    ):
        self.tensorrt_config = tensorrt_config or TensorRTConfig()
        self.triton_config = triton_config or TritonConfig()
        
        self.tensorrt_optimizer = TensorRTOptimizer(self.tensorrt_config)
        self.triton_deployer = TritonModelDeployer(self.triton_config)
        
        logger.info("🚀 Inference Accelerator initialized")
    
    def accelerate_and_deploy(
        self,
        model: torch.nn.Module,
        example_inputs: torch.Tensor,
        model_name: str = "accelerated_model",
        deploy_to_triton: bool = True
    ) -> Dict[str, Any]:
        """
        Complete acceleration and deployment pipeline
        
        Args:
            model: Model to accelerate
            example_inputs: Example inputs for optimization
            model_name: Name for the model
            deploy_to_triton: Whether to deploy to Triton
            
        Returns:
            Dictionary with acceleration results
        """
        results = {
            "model_name": model_name,
            "tensorrt_optimized": False,
            "triton_deployed": False,
            "benchmark_results": {},
            "errors": []
        }
        
        try:
            # Benchmark original model
            logger.info("📊 Benchmarking original model...")
            original_benchmark = self.tensorrt_optimizer.benchmark_model(
                model, example_inputs, num_runs=50
            )
            results["original_benchmark"] = original_benchmark
            
            # TensorRT optimization
            logger.info("⚡ Applying TensorRT optimization...")
            optimized_model = self.tensorrt_optimizer.optimize_model(
                model, example_inputs, model_name
            )
            
            if optimized_model is not None:
                results["tensorrt_optimized"] = True
                
                # Benchmark optimized model
                logger.info("📊 Benchmarking optimized model...")
                optimized_benchmark = self.tensorrt_optimizer.benchmark_model(
                    optimized_model, example_inputs, num_runs=50
                )
                results["optimized_benchmark"] = optimized_benchmark
                
                # Calculate speedup
                if "avg_latency_ms" in original_benchmark and "avg_latency_ms" in optimized_benchmark:
                    speedup = original_benchmark["avg_latency_ms"] / optimized_benchmark["avg_latency_ms"]
                    results["speedup"] = speedup
                    logger.info(f"🎯 Speedup achieved: {speedup:.2f}x")
                
                # Deploy to Triton if requested
                if deploy_to_triton:
                    logger.info("📦 Deploying to Triton...")
                    
                    # Deploy PyTorch version
                    pytorch_deployed = self.triton_deployer.prepare_pytorch_model(
                        model, f"{model_name}_pytorch"
                    )
                    
                    # Save and deploy TensorRT version
                    engine_saved = self.tensorrt_optimizer.save_engine(
                        model_name, f"tensorrt_engines/{model_name}"
                    )
                    
                    if engine_saved:
                        tensorrt_deployed = self.triton_deployer.prepare_tensorrt_model(
                            f"tensorrt_engines/{model_name}/{model_name}.engine",
                            f"{model_name}_tensorrt"
                        )
                        results["triton_deployed"] = pytorch_deployed or tensorrt_deployed
            
            logger.info("✅ Acceleration and deployment pipeline complete")
            return results
            
        except Exception as e:
            error_msg = f"Acceleration pipeline failed: {e}"
            logger.error(f"❌ {error_msg}")
            results["errors"].append(error_msg)
            return results


# Integration functions
def create_tensorrt_optimizer(precision: str = "fp16") -> TensorRTOptimizer:
    """Create TensorRT optimizer with default config"""
    config = TensorRTConfig(precision=precision)
    return TensorRTOptimizer(config)


def create_triton_deployer(model_name: str = "mark_ai") -> TritonModelDeployer:
    """Create Triton deployer with default config"""
    config = TritonConfig(model_name=model_name)
    return TritonModelDeployer(config)


def integrate_with_existing_models():
    """
    Integration function for existing MARK AI models
    
    This can be called to add TensorRT/Triton support to existing models.
    """
    try:
        from src.core.mamba_loader import load_mamba_model
        from src.core.model_registry import get_model_instance
        
        logger.info("🔗 Integrating TensorRT/Triton with existing models")
        
        def accelerate_model(model_key: str = "mamba", precision: str = "fp16"):
            """Accelerate existing model with TensorRT"""
            
            # Load model
            if model_key == "mamba":
                model_instance = load_mamba_model()
                if model_instance.available:
                    model = model_instance.model
                else:
                    logger.warning("Mamba model not available")
                    return None
            else:
                model_instance = get_model_instance(model_key)
                if model_instance:
                    model = model_instance.model
                else:
                    logger.warning(f"Model '{model_key}' not available")
                    return None
            
            # Create accelerator
            accelerator = InferenceAccelerator()
            
            # Create example inputs (adjust based on your model)
            example_inputs = torch.randint(0, 1000, (1, 128), dtype=torch.long)
            
            # Accelerate and deploy
            results = accelerator.accelerate_and_deploy(
                model=model,
                example_inputs=example_inputs,
                model_name=f"mark_ai_{model_key}",
                deploy_to_triton=True
            )
            
            return results
        
        return accelerate_model
        
    except ImportError as e:
        logger.warning(f"⚠️  Could not integrate TensorRT/Triton: {e}")
        return None
