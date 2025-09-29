# Stage 1: Build stage
FROM node:22-alpine AS builder

# Enable corepack for pnpm
RUN corepack enable

# Set working directory
WORKDIR /app

# Copy package files
COPY package.json pnpm-lock.yaml ./

# Install dependencies using pnpm
RUN pnpm install --frozen-lockfile

# Copy source code
COPY . .

# Build the application for production using pnpm
RUN pnpm run build

# Stage 2: Production stage with nginx
FROM nginx:alpine

# Copy built files from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Expose port 80
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
