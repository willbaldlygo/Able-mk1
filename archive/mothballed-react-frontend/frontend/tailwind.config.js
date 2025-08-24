/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Avi sophisticated muted research palette
        primary: {
          50: '#f7f3f4',
          100: '#ede4e6',
          200: '#dcc9ce',
          300: '#c5a2ab',
          400: '#a67b88',
          500: '#8c3041', // Deep burgundy
          600: '#7d2b3a',
          700: '#6b2532',
          800: '#5a202b',
          900: '#4a1b24',
        },
        accent: {
          sage: '#aebfbc', // Sage green for secondary accents
          beige: '#f2e0c9', // Warm beige for backgrounds
          rose: '#f2a7a0', // Dusty rose for highlights
          coral: '#d96c6c', // Coral for active states
        },
        glass: {
          white: 'rgba(255, 250, 244, 0.95)', // Off white base
          light: 'rgba(255, 250, 244, 0.8)',
          dark: 'rgba(48, 33, 39, 0.1)', // Charcoal tint
        },
        text: {
          primary: '#302127', // Charcoal for main text
          secondary: '#8c3041', // Burgundy for emphasis
          muted: '#6b5b5e', // Muted brown-grey
          accent: '#8c3041', // Burgundy for links
          light: '#fffaf4', // Off white for text on dark
        },
        // Keep some blue for system elements
        blue: {
          50: '#eff6ff',
          100: '#dbeafe',
          200: '#bfdbfe',
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
          800: '#1e40af',
          900: '#1e3a8a',
        },
      },
      fontFamily: {
        sans: ['system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
      },
      boxShadow: {
        'card': '0 4px 20px rgba(140, 48, 65, 0.1)',
        'card-hover': '0 8px 32px rgba(140, 48, 65, 0.15)',
        'glass': '0 4px 20px rgba(140, 48, 65, 0.08)',
      },
      borderRadius: {
        'card': '16px',
      },
      backdropBlur: {
        'glass': '10px',
      },
    },
  },
  plugins: [],
}