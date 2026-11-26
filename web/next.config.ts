import type { NextConfig } from "next";

const config: NextConfig = {
  reactStrictMode: true,
  // three.js ships ESM that Next must transpile for the 3D viewer.
  transpilePackages: ["three"],
};

export default config;
