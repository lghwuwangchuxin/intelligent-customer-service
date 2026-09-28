# ---- 构建 ----
FROM node:22-alpine AS build

WORKDIR /app

# 先装依赖再拷源码：改代码不会让依赖层失效
COPY package.json package-lock.json* ./
RUN npm ci || npm install

COPY . .
RUN npm run build

# ---- 运行 ----
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
