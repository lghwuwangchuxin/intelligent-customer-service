import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 3000,
    proxy: {
      // 后端接口在根路径（/chat、/knowledge…）。前端统一走 /api 前缀，
      // 由代理剥掉——这样 dev 与 nginx 生产用同一套路径约定。
      '/api': {
        target: process.env.ICS_API_URL ?? 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ''),
      },
    },
  },
})
