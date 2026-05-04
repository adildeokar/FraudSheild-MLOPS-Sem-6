/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || '',
  },
  async rewrites() {
    const backend = process.env.BACKEND_URL || 'http://localhost:8000';
    return [
      // Primary: same-origin /api/* → FastAPI /api/* (reliable in dev + production)
      {
        source: '/api/:path*',
        destination: `${backend}/api/:path*`,
      },
      // Legacy: /backend/api/... used in older docs
      {
        source: '/backend/:path*',
        destination: `${backend}/:path*`,
      },
    ];
  },
}

module.exports = nextConfig
