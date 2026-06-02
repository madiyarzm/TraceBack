import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  build: {
    // Output to a predictable location so the extension host can reference it
    outDir: 'dist',
    rollupOptions: {
      output: {
        // Single JS and CSS file — VS Code webviews work best with no dynamic imports
        entryFileNames: 'assets/index.js',
        chunkFileNames: 'assets/index.js',
        assetFileNames: 'assets/index.css',
      },
    },
  },
  // Required for the webview to work without a real server — assets must use relative paths
  base: './',
});
