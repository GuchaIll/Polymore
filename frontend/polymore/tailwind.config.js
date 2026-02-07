/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Dark mode colors - cyan panels, black/gray editor
        'poly-bg': '#0a0a0c',
        'poly-sidebar': '#0c2a2a',
        'poly-card': '#0f3d3d',
        'poly-border': '#1a5555',
        'poly-accent': '#22d3ee',
        'poly-danger': '#f59e0b',
        'poly-text': '#ffffff',
        'poly-muted': '#a1a1aa',
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