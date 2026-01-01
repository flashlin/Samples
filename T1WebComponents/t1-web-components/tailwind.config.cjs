/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
        "./demo/index.html",
        "./demo/**/*.{vue,js,ts,jsx,tsx}",
        "./src/lib/**/*.{vue,js,ts,jsx,tsx}",
    ],
    darkMode: 'class',
    theme: {
        extend: {
            colors: {
                dark: {
                    primary: '#1f2937',
                    secondary: '#374151',
                    accent: '#4b5563',
                }
            }
        },
    },
    plugins: [],
}
