import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  base: "/particlelife-sim/",
  // Pin `motion/react` pre-bundle so dev HMR does not 504 "Outdated Optimize Dep"
  // when deps change or the browser holds stale optimized chunk URLs.
  optimizeDeps: {
    include: ["motion/react"],
  },
  build: {
    outDir: "docs",
    emptyOutDir: true,
  },
})
