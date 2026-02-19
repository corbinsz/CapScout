import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
      },
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('maplibre-gl') || id.includes('react-map-gl')) {
            return 'maplibre'
          }
          if (id.includes('recharts') || id.includes('d3-')) {
            return 'recharts'
          }
          if (id.includes('node_modules/react/') || id.includes('node_modules/react-dom/') ||
              id.includes('react-router') || id.includes('node_modules/axios/')) {
            return 'vendor'
          }
        },
      },
    },
  },
})
