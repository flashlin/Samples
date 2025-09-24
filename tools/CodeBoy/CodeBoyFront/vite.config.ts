import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { viteMockServe } from 'vite-plugin-mock'
import path from 'path'

// https://vite.dev/config/
export default defineConfig(({ command }) => {
  const isDev = command === 'serve'
  
  return {
    plugins: [
      vue(),
      // Only enable mock when not using real API
      viteMockServe({
        mockPath: 'mock',
        enable: isDev && !process.env.VITE_USE_REAL_API, // Disable mock when using real API
      }),
    ],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    server: {
      proxy: {
        // Proxy API requests to backend server
        '/api': {
          target: 'http://127.0.0.1:8080',
          changeOrigin: true,
          secure: false,
          configure: (proxy, options) => {
            proxy.on('error', (err, _req, _res) => {
              console.log('Proxy error:', err);
            });
            proxy.on('proxyReq', (_proxyReq, req, _res) => {
              console.log('Proxying request:', req.method, req.url, '-> target:', options.target + (req.url || ''));
            });
            proxy.on('proxyRes', (proxyRes, req, _res) => {
              console.log('Received response:', proxyRes.statusCode, req.url);
            });
          }
        }
      }
    }
  }
})
