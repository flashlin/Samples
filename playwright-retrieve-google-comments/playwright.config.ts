import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
   timeout: 10 * 60 * 1000,
   testDir: 'tests',
   fullyParallel: true,
   forbidOnly: !!process.env.CI,
   retries: process.env.CI ? 2 : 0,
   workers: process.env.CI ? 1 : undefined,
   reporter: 'html',
   use: {
      baseURL: 'http://127.0.0.1:5005',
      trace: 'on-first-retry',
      headless: false,
   },
   projects: [
      {
         name: 'chromium',
         use: { ...devices['Desktop Chrome'] },
      },
   ],
});