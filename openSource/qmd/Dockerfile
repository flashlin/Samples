FROM node:22-bookworm

WORKDIR /app

# Install build dependencies for llama.cpp compilation
RUN apt-get update && apt-get install -y cmake build-essential python3 && rm -rf /var/lib/apt/lists/*

# Install tsx and bun globally
RUN npm install -g tsx bun

# Copy configuration files
COPY package.json bun.lock tsconfig.json tsconfig.build.json ./

# Install dependencies
RUN bun install

# Copy source code
COPY . .

# Ensure cache directories exist for models and index
RUN mkdir -p /root/.cache/qmd/models /documents

# Expose HTTP port in case users want to use --http
EXPOSE 8181

# qmd requires mapping documents to /documents to build the index
# It also uses /root/.cache/qmd/models to store and load AI models.
ENTRYPOINT ["/bin/sh", "-c", "tsx src/qmd.ts collection add /documents && tsx src/qmd.ts embed && exec tsx src/qmd.ts mcp --http --port 8181 --host 0.0.0.0"]
