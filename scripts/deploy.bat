@echo off
echo 🦅 Pushing Argus V3.1 to Server...
scp dashboard.py root@159.65.248.106:/opt/argus/dashboard.py

echo 🔄 Restarting Dashboard Service...
ssh root@159.65.248.106 "sudo systemctl restart dashboard"

echo ✅ Deploy Complete!
pause