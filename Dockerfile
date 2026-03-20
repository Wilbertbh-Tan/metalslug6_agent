# Metal Slug RL training in a container with a virtual desktop.
# Run RetroArch and training inside the same container so host keyboard/focus are unaffected.
# NVIDIA CUDA runtime base gives us GPU access for PyTorch when run with --gpus all.
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:99
ENV PYTHONUNBUFFERED=1
ENV LIBGL_ALWAYS_SOFTWARE=1

# System packages: virtual desktop, build tools, Mesa, noVNC.
RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    x11vnc \
    openbox \
    xauth \
    util-linux \
    xdotool \
    wmctrl \
    x11-utils \
    python3 \
    python3-pip \
    python3-venv \
    python3-tk \
    python3-dev \
    build-essential \
    git \
    cmake \
    pkg-config \
    libgl-dev \
    libsdl2-dev \
    libgl1-mesa-dri \
    libgl1-mesa-glx \
    libglu1-mesa \
    libx11-xcb-dev \
    libxkbcommon-dev \
    libwayland-dev \
    libegl-dev \
    novnc \
    websockify \
    && rm -rf /var/lib/apt/lists/*

# Build RetroArch from source with full desktop OpenGL (not GLES).
# The Ubuntu arm64 package only has GLES, which breaks cores like Flycast.
RUN git clone --depth 1 https://github.com/libretro/RetroArch.git /tmp/RetroArch \
    && cd /tmp/RetroArch \
    && ./configure --enable-opengl --disable-opengles --disable-vulkan \
       --disable-pulse --disable-oss --disable-alsa --disable-jack \
       --disable-wayland --enable-x11 --enable-sdl2 \
    && make -j"$(nproc)" \
    && make install \
    && cd / && rm -rf /tmp/RetroArch

# Compile FBNeo libretro core from source (works on both x86_64 and aarch64).
RUN git clone --depth 1 https://github.com/libretro/FBNeo.git /tmp/FBNeo \
    && make -C /tmp/FBNeo/src/burner/libretro -j"$(nproc)" \
    && mkdir -p /usr/lib/libretro \
    && cp /tmp/FBNeo/src/burner/libretro/fbneo_libretro.so /usr/lib/libretro/fbneo_libretro.so \
    && rm -rf /tmp/FBNeo

# Compile Flycast libretro core (Dreamcast/Atomiswave/Naomi — needed for Metal Slug 6).
RUN git clone --recursive --depth 1 https://github.com/flyinghead/flycast.git /tmp/flycast \
    && cmake -B /tmp/flycast/build -S /tmp/flycast \
       -DLIBRETRO=ON -DCMAKE_BUILD_TYPE=Release \
       -DUSE_VULKAN=OFF \
    && cmake --build /tmp/flycast/build --target flycast_libretro -j"$(nproc)" \
    && cp /tmp/flycast/build/flycast_libretro.so /usr/lib/libretro/flycast_libretro.so \
    && rm -rf /tmp/flycast

# Bake headless RetroArch config into the image.
COPY config/retroarch-container.cfg /etc/retroarch.cfg

WORKDIR /app

# Python deps (opencv-python-headless for no GUI in container)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . .

# Default: start virtual desktop then run training. Override CMD for custom workflows.
RUN chmod +x /app/scripts/run_virtual_desktop.sh /app/scripts/test_headless_container.sh
CMD ["/app/scripts/run_virtual_desktop.sh"]
