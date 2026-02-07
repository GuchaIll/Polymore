# Stage 1: Build the React application
FROM oven/bun:1 as build
WORKDIR /app
COPY package*.json ./
RUN bun install --legacy-peer-deps
COPY . .
RUN bun run build

# Stage 2: Serve with Nginx
FROM nginx:alpine
# Copy the built assets from Stage 1 to Nginx
COPY --from=build /app/build /usr/share/nginx/html
# Copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf
# Expose port 80
EXPOSE 80
# Start Nginx
CMD ["nginx", "-g", "daemon off;"]