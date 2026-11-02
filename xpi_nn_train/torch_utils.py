import torch

DEVICES = ['auto', 'cuda', 'rocm', 'cpu']
DTYPES = {
  'fp32': torch.float32,
  'fp16': torch.float16,
  'bf16': torch.bfloat16,
}


def resolve_dtype(dtype:str) -> torch.dtype:
  return DTYPES.get(dtype, dtype)


def resolve_device(device:str) -> str:
  if device == 'auto':
    return 'cuda' if torch.cuda.is_available() else 'cpu'
  return device


def torch_backends_info():
  # https://pytorch.ac.cn/docs/2.7/backends.html

  print('[versions]')
  print('  torch:', torch.__version__)
  try:
    import torchvision
    print('  torchvision:', torchvision.__version__)
  except ImportError:
    pass
  try:
    import torchaudio
    print('  torchaudio:', torchaudio.__version__)
  except ImportError:
    pass
  print()

  print('[threads]')
  print('  get_num_threads:', torch.get_num_threads())
  print('  get_num_interop_threads:', torch.get_num_interop_threads())

  print('[cpu]')
  print('  get_cpu_capability:', torch.backends.cpu.get_cpu_capability())

  print('[cuda]')
  print('  is_built:', torch.backends.cuda.is_built())
  print('  is_available:', torch.cuda.is_available())
  print('[cuda-device]')
  print('  get_arch_list:', torch.cuda.get_arch_list())
  print('  device_count:', torch.cuda.device_count())
  print('  get_device_properties:', torch.cuda.get_device_properties())
  print('  get_device_name:', torch.cuda.get_device_name())
  print('  get_device_capability:', torch.cuda.get_device_capability())
  print('[cuda-dtype]')
  print('  is_bf16_supported:', torch.cuda.is_bf16_supported())
  print('  is_tf32_supported:', torch.cuda.is_tf32_supported())
  print('  fp32_precision:', torch.backends.cuda.matmul.fp32_precision)
  print('  allow_tf32:', torch.backends.cuda.matmul.allow_tf32)
  print('  allow_fp16_accumulation:', torch.backends.cuda.matmul.allow_fp16_accumulation)
  print('  allow_fp16_reduced_precision_reduction:', torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
  print('  allow_bf16_reduced_precision_reduction:', torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
  print('[cuda-sdp]')
  print('  math_sdp_enabled:', torch.backends.cuda.math_sdp_enabled())  # 通常是debug用
  print('  cudnn_sdp_enabled:', torch.backends.cuda.cudnn_sdp_enabled())
  print('  flash_sdp_enabled:', torch.backends.cuda.flash_sdp_enabled())
  print('  mem_efficient_sdp_enabled:', torch.backends.cuda.mem_efficient_sdp_enabled())
  print('  is_flash_attention_available:', torch.backends.cuda.is_flash_attention_available())
  print('  fp16_bf16_reduction_math_sdp_allowed:', torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed())
  print('[cuda-memory]')
  #print('  memory_summary:')
  #print(torch.cuda.memory.memory_summary())
  #print('  memory_stats:', torch.cuda.memory.memory_stats())
  print('  mem_get_info:', torch.cuda.memory.mem_get_info())
  print('  memory_allocated:', torch.cuda.memory.memory_allocated())
  print('  memory_reserved:', torch.cuda.memory.memory_reserved())
  print('  max_memory_allocated:', torch.cuda.memory.max_memory_allocated())
  print('  max_memory_reserved:', torch.cuda.memory.max_memory_reserved())
  if torch.cuda._HAS_PYNVML or torch.cuda._HAS_AMDSMI:
    # require package 'nvidia-ml-py'
    print('[cuda-status]')
    print('  device_memory_used:', torch.cuda.device_memory_used())
    print('  memory_usage:', torch.cuda.memory_usage())
    print('  utilization:', torch.cuda.utilization())
    print('  temperature:', torch.cuda.temperature())
    print('  power_draw:', torch.cuda.power_draw())
    print('  clock_rate:', torch.cuda.clock_rate())
  print('[cudnn]')
  print('  is_available:', torch.backends.cudnn.is_available())
  print('  enabled:', torch.backends.cudnn.enabled)
  print('  version:', torch.backends.cudnn.version())
  print('  allow_tf32:', torch.backends.cudnn.allow_tf32)
  print('  deterministic:', torch.backends.cudnn.deterministic)
  print('  benchmark:', torch.backends.cudnn.benchmark)
  print('  benchmark_limit:', torch.backends.cudnn.benchmark_limit)

  print('[mps]')
  print('  is_built:', torch.backends.mps.is_built())
  print('  is_available:', torch.backends.mps.is_available())
  print('[mkl]')
  print('  is_available:', torch.backends.mkl.is_available())
  print('[mkldnn]')
  print('  is_available:', torch.backends.mkldnn.is_available())
  print('  is_acl_available:', torch.backends.mkldnn.is_acl_available())
  print('[openmp]')
  print('  is_available:', torch.backends.openmp.is_available())
  print('[opt_einsum]')
  print('  is_available:', torch.backends.opt_einsum.is_available())
  print('[nnpack]')
  print('  is_available:', torch.backends.nnpack.is_available())
