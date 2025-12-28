import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    allowedHosts: ['rpi5kavecany.local', 'localhost', '192.168.1.125'],
    proxy: {
      '/api': {
        target: 'http://localhost:8765',
        changeOrigin: true,
      },
      '/stream': {
        target: 'http://localhost:8765',
        changeOrigin: true,
      },
    },
  },
})


