import type { Config } from "tailwindcss"

const config: Config = {
  darkMode: ["class"],
  content: [
    './pages/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
    './app/**/*.{ts,tsx}',
    './src/**/*.{ts,tsx}',
  ],
  prefix: "",
  theme: {
    container: {
      center: true,
      padding: "2rem",
      screens: {
        "2xl": "1400px",
      },
    },
    extend: {
      colors: {
        border: "#e6e1db",
        input: "#e6e1db",
        ring: "#6400FF",
        background: "#f8f9fa",
        foreground: "#000000",
        primary: {
          DEFAULT: "#6400FF",
          foreground: "#ffffff",
        },
        secondary: {
          DEFAULT: "#f8f9fa",
          foreground: "#000000",
        },
        destructive: {
          DEFAULT: "#C61C00",
          foreground: "#ffffff",
        },
        muted: {
          DEFAULT: "#f8f9fa",
          foreground: "#6b7280",
        },
        accent: {
          DEFAULT: "#f8f9fa",
          foreground: "#000000",
        },
        popover: {
          DEFAULT: "#ffffff",
          foreground: "#000000",
        },
        card: {
          DEFAULT: "#ffffff",
          foreground: "#000000",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      keyframes: {
        "accordion-down": {
          from: { height: "0" },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: "0" },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
      },
      fontFamily: {
        'heading': ['Hanken Grotesk', 'system-ui', '-apple-system', 'sans-serif'],
        'body': ['Neuton', 'Georgia', 'Times New Roman', 'serif'],
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
} satisfies Config

export default config
