import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5174,
    host: true,   // bind to 0.0.0.0 — required for LAN / tunnel access
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8001',   // UnslothCraft backend
        changeOrigin: true,
      },
      '/v1': {
        target: 'http://127.0.0.1:8001',   // UnslothCraft backend (OpenAI-compat)
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          recharts: ['recharts'],
          motion: ['framer-motion'],
        },
      },
    },
  },
})
