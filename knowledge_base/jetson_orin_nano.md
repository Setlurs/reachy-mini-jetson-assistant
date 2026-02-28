NVIDIA Jetson is a family of embedded AI computing platforms for edge AI, robotics, and autonomous machines. Founded in 1993 by Jensen Huang, NVIDIA is headquartered in Santa Clara, California. The Jetson lineup includes Orin Nano with 67 TOPS and 8GB RAM at 249 dollars, Orin NX with 70 to 100 TOPS and 8 to 16GB RAM, and AGX Orin with up to 275 TOPS and 32 or 64GB RAM. All share the same JetPack SDK. A 4GB Orin Nano variant also exists with reduced specs.

The Jetson Orin Nano delivers 67 INT8 TOPS of AI performance. It has an NVIDIA Ampere GPU with 1024 CUDA cores and 32 Tensor cores, a 6-core Arm Cortex A78AE CPU at 1.7 GHz with 1.5 megabytes L2 and 4 megabytes L3 cache, and 8GB 128-bit LPDDR5 unified memory at 102 gigabytes per second bandwidth shared between CPU and GPU. Module size is 70 by 45 millimeters.

The Developer Kit costs 249 dollars and includes Gigabit Ethernet, four USB 3.2 Gen 2 ports at 10 gigabits per second, USB-C, DisplayPort 1.2 supporting 4K at 30 hertz, HDMI 1.4, two MIPI CSI-2 camera connectors supporting up to 4 cameras, two M.2 Key M slots for NVMe SSD, one M.2 Key E slot for WiFi and Bluetooth, a 40-pin GPIO header compatible with Raspberry Pi HATs, and a microSD card slot. Operating temperature range is minus 25 to 90 degrees Celsius. Power is provided through USB-C at 5 volts or a DC barrel jack. It supports desktop mode with a monitor or headless operation over SSH.

It has three power modes: 7 watts for battery use, 15 watts for balanced performance, and 25 watts for maximum performance. You can set the power mode using the nvpmodel command. The developer kit includes an active cooling fan.

It runs AI models locally without internet, making it ideal for privacy-sensitive applications and remote deployments. Applications include large language models like Llama, Gemma, and Mistral, object detection with YOLO and ResNet, speech recognition and text-to-speech for voice assistants, ROS2 robotics, smart camera analytics, vision-language models, industrial IoT, and autonomous drones.


Compatible cameras include MIPI CSI-2 types such as Raspberry Pi Camera version 2, Arducam IMX477 and IMX219, and Leopard Imaging cameras. It also supports USB cameras via USB 3.2 ports and IP cameras via Ethernet.

It runs JetPack 6 based on Ubuntu 22.04 with CUDA 12, cuDNN, TensorRT, OpenCV, VPI, and NVIDIA Container Runtime pre-installed. It boots from microSD or NVMe SSD.

Supported AI frameworks include PyTorch, TensorFlow, ONNX Runtime, TensorRT which provides 2 to 5 times inference speedup, llama.cpp for running quantized large language models, faster-whisper for faster speech-to-text, and Piper TTS for local text-to-speech. NVIDIA tools include Isaac for ROS2 robotics, DeepStream for video analytics, TAO for transfer learning, and Triton for model serving.

Docker is fully supported with GPU passthrough. Pre-built containers are available from NGC and the jetson-containers GitHub project, including llama.cpp, Ollama, Whisper, Stable Diffusion, and robotics stacks.

For system monitoring, use jtop or tegrastats. Note that nvidia-smi is not available on Jetson. For best performance, use quantized INT4 or INT8 models, TensorRT optimization, and an NVMe SSD which is 5 to 10 times faster than microSD. A 256 gigabyte or larger NVMe SSD is recommended since models and containers can take 10 to 50 gigabytes.

For networking, Ethernet is most reliable. WiFi requires an M.2 Key E module. You can connect remotely via SSH.

To get started, flash JetPack to a microSD card using SDK Manager, connect a monitor and Ethernet cable, complete the Ubuntu setup, update the system, install jetson-stats for monitoring, set maximum performance mode, optionally add an NVMe SSD, and then start developing.
