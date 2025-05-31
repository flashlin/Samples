module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  safelist: [
    ...Array.from({ length: 50 }, (_, i) => `grid-cols-${i + 1}`),
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}