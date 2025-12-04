#!/usr/bin/env bash

# Update package listings
apt-get update

# Install root certificates
apt-get install -y ca-certificates

# Install Python dependencies
pip install -r requirements.txt