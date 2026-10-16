FROM debian:sid-slim
RUN apt-get update && apt-get install -y --no-install-recommends mesa-vulkan-drivers libvulkan1 vulkan-tools ca-certificates libgomp1 libcurl4 && rm -rf /var/lib/apt/lists/* && apt-cache policy mesa-vulkan-drivers | head -3
