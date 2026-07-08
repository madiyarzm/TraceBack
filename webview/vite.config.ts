import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  build: {
    // Output to a predictable location so the extension host can reference it
    outDir: 'dist',
    // One CSS bundle — the provider links exactly assets/index.css
    cssCodeSplit: false,
    rollupOptions: {
      output: {
        // Single JS and CSS file — VS Code webviews work best with no dynamic imports.
        // Only stylesheets may take the index.css name: binary assets (bundled
        // woff2 fonts) keep their own names, otherwise a font wins the name
        // collision and the webview loads a woff2 as its stylesheet.
        entryFileNames: 'assets/index.js',
        chunkFileNames: 'assets/index.js',
        assetFileNames: (info) =>
          info.name?.endsWith('.css') ? 'assets/index.css' : 'assets/[name]-[hash][extname]',
      },
    },
  },
  // Required for the webview to work without a real server — assets must use relative paths
  base: './',
});
