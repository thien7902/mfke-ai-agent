# Use ubuntu as base image.
# We need Python 3.12 because graphiti-core's `falkordblite` extra
# (embedded FalkorDB via redislite — what MemoryService uses in
# GRAPHITI_FALKOR_MODE=embedded default) requires Python >= 3.12.
# The previous ubuntu:22.04 + python3 (3.10) build silently skipped
# the falkordblite extra and AsyncFalkorDB wasn't importable.
FROM ubuntu:24.04

LABEL description="Telegram Bot with HolmesGPT, MongoDB, and all required tools"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install additional Python packages and tools.
# Acquire options below work around clock-drift against Ubuntu's
# NotValidYet-signed InRelease files (builder clock vs repo signing time):
# `Check-Valid-Until=false` skips "expired" warnings, and
# `Check-Date=false` skips "not valid yet" (builder clock in the future).
RUN apt-get update \
        -o Acquire::Check-Valid-Until="false" \
        -o Acquire::Check-Date="false" && \
    apt-get install -y --no-install-recommends \
        bat \
        bsdextrautils \
        ca-certificates \
        curl \
        fzf \
        gcc \
        gpg \
        jq \
        less \
        libssl-dev \
        parallel \
        python3 \
        python3-pip \
        python3-venv \
        ripgrep \
        procps \
        ttyd \
        tzdata \
        tesseract-ocr \
        tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install pipx.
# Ubuntu 24.04 marks the system Python as externally-managed (PEP 668).
# In a container with no other Python users, --break-system-packages is
# safe and matches how the old ubuntu:22.04 image behaved.
ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN pip install --no-cache-dir --ignore-installed --upgrade pip setuptools wheel pipx

# Install OpenStack CLIs and HolmesGPT via pip
RUN pip install --no-cache-dir \
        python-openstackclient \
        python-octaviaclient \
        holmesgpt

# Install kubectl
RUN curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.31/deb/Release.key | gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.31/deb/ /" > /etc/apt/sources.list.d/kubernetes.list && \
    apt-get update \
        -o Acquire::Check-Valid-Until="false" \
        -o Acquire::Check-Date="false" && \
    apt-get install -y --no-install-recommends kubectl && \
    rm -rf /var/lib/apt/lists/*

# Create .holmes config directory for HolmesGPT
RUN mkdir -p /root/.holmes

# Copy entrypoint script
COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY .env.example .env.example

# Verify installations
RUN holmes --help > /dev/null && echo "HolmesGPT verified"

# Expose port for health checks
EXPOSE 8080

# Health check for Docker
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/live || exit 1

# Use entrypoint script
ENTRYPOINT ["/entrypoint.sh"]
CMD ["default", "3"]