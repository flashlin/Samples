# 建構階段
FROM node:20-alpine AS builder

# 安裝 pnpm (使用 corepack 更有效率)
RUN corepack enable

# 設定工作目錄
WORKDIR /app

# 設定 pnpm 環境變數優化
ENV PNPM_HOME="/pnpm"
ENV PATH="$PNPM_HOME:$PATH"

# 複製 pnpm 相關檔案 (分層快取優化)
COPY package.json pnpm-lock.yaml ./

# 先複製 scripts 資料夾（如果 postinstall 需要）
COPY scripts/ ./scripts/
COPY public/ ./public/

# 設定 pnpm store 在 Docker layer 中快取
RUN --mount=type=cache,id=pnpm,target=/pnpm/store pnpm install --frozen-lockfile

# 複製源碼
COPY . .

# 構建生產版本
RUN pnpm run build

# 生產階段 - 使用 nginx alpine
FROM nginx:alpine

# 移除 nginx 預設內容
RUN rm -rf /usr/share/nginx/html/*

# 複製構建好的檔案
COPY --from=builder /app/dist /usr/share/nginx/html

# 創建 dist 目錄並移動除了 index.html 外的所有檔案
RUN mkdir -p /usr/share/nginx/html/dist && \
    cd /usr/share/nginx/html && \
    find . -maxdepth 1 ! -name '.' ! -name 'index.html' ! -name 'dist' -exec mv {} dist/ \;

# 複製 nginx 配置
COPY nginx.conf /etc/nginx/nginx.conf

# 暴露 80 端口
EXPOSE 80

# 啟動 nginx
CMD ["nginx", "-g", "daemon off;"] 