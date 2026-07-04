def select_most_free_gpu(cuda_device_count, nvml_init, nvml_get_handle, nvml_get_memory):
    """Return the visible GPU index with the lowest used memory."""
    nvml_init()
    device_count = int(cuda_device_count())
    if device_count <= 0:
        return 0

    selected_gpu = 0
    min_mem_used = float("inf")
    for gpu_id in range(device_count):
        try:
            handle = nvml_get_handle(gpu_id)
            meminfo = nvml_get_memory(handle)
            mem_used = meminfo.used
        except Exception:
            continue
        if mem_used < min_mem_used:
            min_mem_used = mem_used
            selected_gpu = gpu_id
    return selected_gpu
