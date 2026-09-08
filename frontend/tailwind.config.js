/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#eef6ff",
          100: "#d9ecff",
          200: "#bcdcff",
          300: "#8ec4ff",
          400: "#59a3ff",
          500: "#3182f6",
          600: "#1f63e0",
          700: "#1a4fb4",
          800: "#1b4291",
          900: "#1c3873",
        },
      },
    },
  },
  plugins: [],
};
