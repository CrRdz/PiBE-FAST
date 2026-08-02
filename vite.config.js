import {defineConfig} from 'vite';
import {resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const projectRoot = fileURLToPath(new URL('.', import.meta.url));

export default defineConfig({
  build: {
    outDir: resolve(projectRoot, 'app/static/react'),
    emptyOutDir: true,
    rollupOptions: {
      input: resolve(projectRoot, 'frontend/main.jsx'),
      output: {
        entryFileNames: 'app.js',
        chunkFileNames: 'chunks/[name].js',
        assetFileNames: 'assets/[name][extname]'
      }
    }
  }
});
