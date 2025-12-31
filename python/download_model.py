"""Download DA3-GIANT model."""
import os
import sys

# Disable xet transfer for compatibility
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

# Setup logging to file
log_file = open('download_log.txt', 'w')

def log(msg):
    print(msg, flush=True)
    log_file.write(msg + '\n')
    log_file.flush()

log('Starting model download...')

try:
    import huggingface_hub
    log(f'huggingface_hub version: {huggingface_hub.__version__}')
    
    # Enable verbose logging
    huggingface_hub.logging.set_verbosity_debug()
    
    from huggingface_hub import hf_hub_download
    
    log('Downloading depth-anything/DA3-GIANT model.safetensors...')
    log('This is a 5.4GB file and may take several minutes.')
    
    path = hf_hub_download(
        repo_id='depth-anything/DA3-GIANT',
        filename='model.safetensors',
        resume_download=True,
        force_download=False,
    )
    
    log(f'SUCCESS! Model downloaded to: {path}')
    
except Exception as e:
    log(f'ERROR: {e}')
    import traceback
    log(traceback.format_exc())
    sys.exit(1)
finally:
    log_file.close()
