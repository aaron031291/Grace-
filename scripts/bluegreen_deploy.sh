#!/bin/bash
# bluegreen_deploy.sh: Blue/Green deploy with shadow traffic
# Usage: ./bluegreen_deploy.sh [blue|green]

set -e
TARGET=$1
if [[ "$TARGET" != "blue" && "$TARGET" != "green" ]]; then
  echo "Usage: $0 [blue|green]"
  exit 1
fi

# Deploy both blue and green containers
# (Assumes docker-compose.bluegreen.yml defines blue_api and green_api services)
docker-compose -f docker-compose.bluegreen.yml up -d blue_api green_api

# Update NGINX config for shadow traffic
cp infra/nginx-bluegreen.conf /etc/nginx/nginx.conf
nginx -s reload

# Cutover: switch main traffic to target
if [[ "$TARGET" == "blue" ]]; then
  sed -i 's/proxy_pass http:\/\/grace_green;/proxy_pass http:\/\/grace_blue;/' /etc/nginx/nginx.conf
else
  sed -i 's/proxy_pass http:\/\/grace_blue;/proxy_pass http:\/\/grace_green;/' /etc/nginx/nginx.conf
fi
nginx -s reload

echo "Blue/Green deploy complete. Main traffic: $TARGET, shadow traffic: $( [[ "$TARGET" == "blue" ]] && echo green || echo blue )"
