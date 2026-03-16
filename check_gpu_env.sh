#!/bin/bash
export LD_LIBRARY_PATH=$(ls -d /usr/local/lib/python3.9/dist-packages/nvidia/*/lib | paste -sd : -):${LD_LIBRARY_PATH}
echo "LD_LIBRARY_PATH is: $LD_LIBRARY_PATH"
python3.9 -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}'); print(f'GPUs found: {tf.config.list_physical_devices(\"GPU\")} ')"
