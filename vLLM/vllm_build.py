from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import docker
from typing import Optional
import os 

app = FastAPI()

class EngineArgs(BaseModel):
    HUGGING_FACE_HUB_TOKEN: str
    model: str = Field(default="facebook/opt-125m")
    tokenizer: Optional[str] = None
    skip_tokenizer_init: bool = Field(default=False)
    revision: Optional[str] = None
    code_revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    tokenizer_mode: str = Field(default="auto", choices=["auto", "slow"])
    trust_remote_code: bool = Field(default=False)
    download_dir: Optional[str] = None
    load_format: str = Field(default="auto", choices=["auto", "pt", "safetensors", "npcache", "dummy", "tensorizer", "bitsandbytes"])
    dtype: str = Field(default="auto", choices=["auto", "half", "float16", "bfloat16", "float", "float32"])
    kv_cache_dtype: str = Field(default="auto", choices=["auto", "fp8", "fp8_e5m2", "fp8_e4m3"])
    quantization_param_path: Optional[str] = None
    max_model_len: Optional[int] = Field(default=1024)
    guided_decoding_backend: str = Field(default="outlines", choices=["outlines", "lm-format-enforcer"])
    distributed_executor_backend: str = Field(default="mp", choices=["ray", "mp"])
    worker_use_ray: bool = Field(default=False)
    pipeline_parallel_size: int = Field(default=1)
    tensor_parallel_size: int = Field(default=1)
    max_parallel_loading_workers: Optional[int] = Field(default=1)
    ray_workers_use_nsight: bool = Field(default=False)
    block_size: int = Field(default=16, choices=[8, 16, 32])
    enable_prefix_caching: bool = Field(default=False)
    disable_sliding_window: bool = Field(default=False)
    use_v2_block_manager: bool = Field(default=False)
    num_lookahead_slots: int = Field(default=0)
    seed: int = Field(default=0)
    swap_space: int = Field(default=4)
    gpu_memory_utilization: float = Field(default=0.9)
    num_gpu_blocks_override: Optional[int] = Field(default=1)
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: int = Field(default=256)
    max_logprobs: int = Field(default=5)
    disable_log_stats: bool = Field(default=False)
    quantization: Optional[str] = Field(default=None, choices=["aqlm", "awq", "deepspeedfp", "fp8", "marlin", "gptq_marlin_24", "gptq_marlin", "gptq", "squeezellm", "sparseml", "bitsandbytes", "None"])
    rope_scaling: Optional[str] = None
    enforce_eager: bool = Field(default=False)
    max_context_len_to_capture: Optional[int] = None
    max_seq_len_to_capture: int = Field(default=8192)
    disable_custom_all_reduce: bool = Field(default=False)
    tokenizer_pool_size: int = Field(default=0)
    tokenizer_pool_type: str = Field(default="ray")
    tokenizer_pool_extra_config: Optional[str] = None
    enable_lora: bool = Field(default=False)
    max_loras: int = Field(default=1)
    max_lora_rank: int = Field(default=16)
    lora_extra_vocab_size: int = Field(default=256)
    lora_dtype: str = Field(default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    long_lora_scaling_factors: Optional[str] = None
    max_cpu_loras: Optional[int] = None
    fully_sharded_loras: bool = Field(default=False)
    device: str = Field(default="auto", choices=["auto", "cuda", "neuron", "cpu"])
    image_input_type: Optional[str] = Field(default=None, choices=["pixel_values", "image_features"])
    image_token_id: Optional[int] = None
    image_input_shape: Optional[str] = None
    image_feature_size: Optional[int] = None
    image_processor: Optional[str] = None
    image_processor_revision: Optional[str] = None
    disable_image_processor: bool = Field(default=False)
    scheduler_delay_factor: float = Field(default=0.0)
    enable_chunked_prefill: bool = Field(default=False)
    speculative_model: Optional[str] = None
    num_speculative_tokens: Optional[int] = None
    speculative_max_model_len: Optional[int] = None
    speculative_disable_by_batch_size: Optional[int] = None
    ngram_prompt_lookup_max: Optional[int] = None
    ngram_prompt_lookup_min: Optional[int] = None
    model_loader_extra_config: Optional[str] = None
    preemption_mode: Optional[str] = Field(default=None, choices=["recompute", "swap"])
    served_model_name: Optional[str] = None

@app.post("/run-docker")
def run_docker(engine_args: EngineArgs):
    try:
        client = docker.from_env()
        home_directory = os.path.expanduser("~")
        volume_path = os.path.join(home_directory, ".cache/huggingface")
        volumes = {volume_path: {'bind': '/root/.cache/huggingface', 'mode': 'rw'}}
        env_vars = {'HUGGING_FACE_HUB_TOKEN': engine_args.HUGGING_FACE_HUB_TOKEN}  # Add other necessary environment variables here
        ports = {'8000/tcp': 8001}

       
        container = client.containers.run(
                'vllm/vllm-openai:latest',
                command=f"--model {engine_args.model} --tokenizer {engine_args.tokenizer}",
                runtime='nvidia',
                volumes=volumes,
                environment=env_vars,
                ports=ports,
                ipc_mode='host',
                detach=True
        )
        
        return {"message": "Container started successfully", "container_id": container.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
