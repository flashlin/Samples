// Copy wa-sqlite.wasm to public/ (cross-platform)
import { copyFile } from 'fs/promises';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const src = resolve(__dirname, '../node_modules/wa-sqlite/dist/wa-sqlite.wasm');
const dest = resolve(__dirname, '../public/wa-sqlite.wasm');

async function copyWasm() {
  try {
    await copyFile(src, dest);
    console.log('wa-sqlite.wasm copied to public/');
  } catch (err) {
    console.error('Failed to copy wa-sqlite.wasm:', err);
    process.exit(1);
  }
}

copyWasm(); 