# Lightweight Production Serving of Vanilla JS Clinical Cockpit
FROM nginx:alpine

# Remove default nginx configurations
RUN rm -rf /etc/nginx/conf.d/*

# Copy custom Nginx configuration on port 3000
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy frontend cockpit assets
COPY public/ /usr/share/nginx/html/

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget -q -O /dev/null http://localhost:3000/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
