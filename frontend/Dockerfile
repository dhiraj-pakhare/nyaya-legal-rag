# ─── Stage 1: Build Frontend Bundle ────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

# Copy dependency specifications for clean, reproducible caching
COPY package.json package-lock.json ./

RUN npm ci

# Copy source tree and configuration files
COPY . .

# Build-time API Base URL argument
ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

# Compile TypeScript and bundle with Vite
RUN npm run build

# ─── Stage 2: Production Nginx Server ─────────────────────────────────────────
FROM nginx:1.25-alpine AS runner

# Clean default html
RUN rm -rf /usr/share/nginx/html/*

# Copy compiled static distribution from builder
COPY --from=builder /app/dist /usr/share/nginx/html

# Apply custom Nginx configuration with SPA routing and SSE reverse proxy
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD wget -q --spider http://127.0.0.1/healthz || exit 1

CMD ["nginx", "-g", "daemon off;"]
