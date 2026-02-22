import { resolve } from 'path';

export default {
  root: resolve(__dirname),
  build: {
    emptyOutDir: false,
    rollupOptions: {
      input: resolve(__dirname, 'static/js/app.js'),
      output: {
        entryFileNames: 'bundle.js',
        chunkFileNames: '[name].js',
        assetFileNames: '[name][extname]',
      },
    },
    outDir: resolve(__dirname, 'static/js'),
  },
};
