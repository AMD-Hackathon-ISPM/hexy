import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { fileURLToPath, URL } from 'node:url'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    proxy: {
      '/health': 'http://localhost:8000',
      '/mujoco': { target: 'http://localhost:8000', ws: true },
      '/assets/rl': 'http://localhost:8000',
      '/assets/cave': {
        target: 'http://localhost:8000',
        proxyTimeout: 120000,
        timeout: 120000,
      },
    },
  },
})
