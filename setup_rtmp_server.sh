#!/bin/bash
# ============================================================================
# Dream Vision - RTMP Server Setup for AWS VM
# ============================================================================
# This script installs and configures an NGINX RTMP server on your AWS VM
# so your laptop can stream webcam video to AWS for processing.
#
# Flow: Laptop Webcam -> RTMP -> AWS (Process) -> RTMP -> Laptop (View)
# ============================================================================

set -e

echo "============================================================"
echo "  Dream Vision - RTMP Streaming Server Setup"
echo "============================================================"
echo ""

# Update system
echo "[1/5] Updating system packages..."
sudo apt-get update -y

# Install NGINX with RTMP module
echo "[2/5] Installing NGINX with RTMP module..."
sudo apt-get install -y libnginx-mod-rtmp nginx ffmpeg

# Configure NGINX RTMP
echo "[3/5] Configuring RTMP server..."

sudo tee /etc/nginx/nginx.conf > /dev/null <<'NGINXCONF'
worker_processes auto;
events {
    worker_connections 1024;
}

# RTMP server for receiving webcam stream
rtmp {
    server {
        listen 1935;
        chunk_size 4096;
        allow publish all;
        allow play all;

        # Receive webcam stream from laptop
        application live {
            live on;
            record off;

            # Low latency settings
            interleave on;
            wait_key on;
            wait_video on;

            # Allow anyone to publish (restrict in production)
            allow publish all;
            allow play all;
        }

        # Output processed stream back to viewer
        application output {
            live on;
            record off;

            interleave on;
            wait_key on;
            wait_video on;

            allow publish all;
            allow play all;
        }
    }
}

# HTTP server for HLS playback (view in browser)
http {
    sendfile off;
    tcp_nopush on;
    directio 512;
    default_type application/octet-stream;

    server {
        listen 8080;

        location /stat {
            rtmp_stat all;
            rtmp_stat_stylesheet stat.xsl;
        }

        location / {
            root /var/www/html;
        }
    }
}
NGINXCONF

# Test NGINX configuration
echo "[4/5] Testing NGINX configuration..."
sudo nginx -t

# Restart NGINX
echo "[5/5] Starting RTMP server..."
sudo systemctl restart nginx
sudo systemctl enable nginx

# Get public IP
PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "<YOUR_AWS_IP>")

echo ""
echo "============================================================"
echo "  RTMP Server Ready!"
echo "============================================================"
echo ""
echo "  Server Status: $(sudo systemctl is-active nginx)"
echo "  RTMP Port:     1935"
echo "  Stats Port:    8080"
echo "  Public IP:     ${PUBLIC_IP}"
echo ""
echo "============================================================"
echo "  IMPORTANT: Open these ports in AWS Security Group!"
echo "============================================================"
echo ""
echo "  Port 1935 (TCP) - RTMP streaming"
echo "  Port 8080 (TCP) - Stats page (optional)"
echo ""
echo "  Go to AWS Console -> EC2 -> Security Groups -> Edit Inbound Rules"
echo "  Add: Custom TCP | Port 1935 | Your IP"
echo "  Add: Custom TCP | Port 8080 | Your IP (optional)"
echo ""
echo "============================================================"
echo "  Next Steps:"
echo "============================================================"
echo ""
echo "  1. Open ports in AWS Security Group (see above)"
echo ""
echo "  2. On your LAPTOP, run:"
echo "     ffmpeg -f dshow -i video=\"YOUR_WEBCAM\" -c:v libx264 -preset ultrafast -tune zerolatency -b:v 2500k -f flv rtmp://${PUBLIC_IP}:1935/live/webcam"
echo ""
echo "  3. On this AWS VM, run:"
echo "     python dream_vision_cloud.py --mode stream --stream-url rtmp://localhost:1935/live/webcam --stream-output rtmp://localhost:1935/output/result"
echo ""
echo "  4. On your LAPTOP, view the result:"
echo "     ffplay rtmp://${PUBLIC_IP}:1935/output/result"
echo ""
echo "============================================================"
