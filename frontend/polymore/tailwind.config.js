/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Dark mode colors
        'poly-bg': '#1a1a2e',
        'poly-sidebar': '#16213e',
        'poly-border': '#0f3460',
        'poly-accent': '#e94560',
        'poly-danger': '#ff6b6b',
        // Light mode colors
        'poly-light-bg': '#f8fafb',
        'poly-light-sidebar': '#ffffff',
        'poly-light-border': '#e2e8f0',
        'poly-light-accent': '#10b981',
        'poly-light-danger': '#ef4444',
        'poly-light-text': '#1e293b',
        'poly-light-muted': '#64748b',
      },
    },
  },
  plugins: [],
}