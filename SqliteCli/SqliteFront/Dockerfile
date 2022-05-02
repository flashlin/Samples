FROM node:lts-slim AS BUILDER
ARG MODE
WORKDIR /app

COPY package.json ./
RUN npm install

COPY . .

RUN npm run build -- --mode ${MODE}

FROM nginx:alpine as RUNTIME
ARG MODE
EXPOSE 80

RUN echo "Setting timezone start..." && \
apk add --no-cache tzdata && \
cp /usr/share/zoneinfo/America/Anguilla /etc/localtime && \
echo "/America/Anguilla" >  /etc/timezone && \
apk del tzdata && \
date && \
echo "Setting timezone end..."

COPY --from=builder /app/nginx/nginx.${MODE}.conf /etc/nginx/nginx.conf
COPY --from=builder /app/dist /usr/share/nginx/html