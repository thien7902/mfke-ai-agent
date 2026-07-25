# Use the provided base image with all required packages
FROM ubuntu:22.04

LABEL description="Telegram Bot with HolmesGPT, MongoDB, and all required tools"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install additional Python packages and tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
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
        ripgrep \
        procps \
        ttyd \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install pipx
RUN pip install --no-cache-dir --upgrade pip setuptools wheel pipx

# Install OpenStack CLIs and HolmesGPT via pip
RUN pip install --no-cache-dir \
        python-openstackclient \
        python-octaviaclient \
        holmesgpt

# Install kubectl
RUN curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.31/deb/Release.key | gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.31/deb/ /" > /etc/apt/sources.list.d/kubernetes.list && \
    apt-get update && \
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

# Expose port for health checks (optional)
EXPOSE 8080

# Use entrypoint script
ENTRYPOINT ["/entrypoint.sh"]
CMD ["default", "3"]