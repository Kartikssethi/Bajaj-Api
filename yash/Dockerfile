# Use official Node.js LTS image
FROM node:20-alpine

# Set working directory
WORKDIR /app

# Install additional packages for large file handling
RUN apk add --no-cache python3 make g++

# Copy package files and install dependencies
COPY package.json pnpm-lock.yaml ./
RUN npm install -g pnpm && pnpm install --frozen-lockfile

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p uploads temp logs faiss-stores

# Expose port (default 3000)
EXPOSE 3000

# Set environment variables for large document handling
ENV NODE_ENV=production
ENV NODE_OPTIONS="--max-old-space-size=6144"

# Start the server with increased memory
CMD ["node", "--max-old-space-size=6144", "server.js"]
