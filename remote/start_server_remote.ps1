# Phase 6 — start the local FastAPI server for remote tunneling.
# Run this script in PowerShell after setting the secure remote environment variables.
#
# Example:
#   $env:SURVEILLANCE_SECURE_COOKIE="1"
#   $env:SURVEILLANCE_SESSION_SECRET="..."
#   $env:SURVEILLANCE_OWNER_PASSWORD="..."
#   $env:SURVEILLANCE_OPERATOR_PASSWORD="..."
#   .\start_server_remote.ps1

python server.py
