# StructureRelations Deployment on Debian

This guide deploys the FastAPI web application as a dedicated systemd service
behind Nginx. It assumes Debian 12 or a compatible Debian-based server and a
checkout of this repository.

## Deployment values

Adjust these values once, then use them consistently:

```text
Application user:  structurerelations
Application root: /opt/StructureRelations
Virtualenv:       /opt/StructureRelations/.venv
Backend address:  127.0.0.1:8101
Public URL:       https://structures.example.org
```

The backend must listen on loopback only. Nginx is the public entry point.

## 1. Install OS packages

Run as an administrator:

```bash
sudo apt update
sudo apt install -y \
    python3 \
    python3-venv \
    python3-dev \
    build-essential \
    graphviz \
    libgraphviz-dev \
    pkg-config \
    unixodbc-dev \
    nginx
```

The application requirements include `pygraphviz` and `pyodbc`, so the
compiler, Graphviz development files, and unixODBC headers are required when
those packages are installed from source.

Verify that the selected Python is 3.11 or newer:

```bash
python3 --version
```

If the server does not provide Python 3.11, install it from the approved
Debian repository for that server before continuing. Do not mix packages from
untrusted repositories into a production host without approval.

## 2. Create the service account and install the application

Create a system account with no interactive shell:

```bash
sudo useradd --system --home /opt/StructureRelations \
    --shell /usr/sbin/nologin structurerelations
sudo mkdir -p /opt/StructureRelations
sudo chown -R structurerelations:structurerelations /opt/StructureRelations
```

Copy or clone the repository into `/opt/StructureRelations`, then set ownership
again. Do not copy `.venv`, `__pycache__`, or development-only output from
another machine.

```bash
sudo chown -R structurerelations:structurerelations /opt/StructureRelations
cd /opt/StructureRelations
sudo -u structurerelations python3 -m venv .venv
sudo -u structurerelations .venv/bin/python -m pip install --upgrade pip
sed '/^python==/d' src/webapp/requirements.txt > /tmp/structurerelations-requirements.txt
sudo -u structurerelations .venv/bin/pip install \
    -r /tmp/structurerelations-requirements.txt
```

The repository requirements file also contains a `python==3.11` environment
pin. Debian provides Python separately, so the install command removes that
non-pip line and installs the remaining application requirements.

Create the persistent session directory explicitly:

```bash
sudo -u structurerelations mkdir -p /opt/StructureRelations/webapp_sessions
```

The application defaults to `webapp_sessions` relative to its working
 directory. The systemd unit below sets the working directory to the
repository root so the path is stable across restarts.

## 3. Test the backend locally

Run this temporary foreground check as the service account:

```bash
cd /opt/StructureRelations
sudo -u structurerelations .venv/bin/python -m uvicorn main:app \
    --app-dir src/webapp \
    --host 127.0.0.1 \
    --port 8101
```

In another terminal, verify the HTTP response:

```bash
curl --fail http://127.0.0.1:8101/
```

Stop Uvicorn with `Ctrl+C` after the check. The browser UI also uses a
WebSocket at `/ws/{session_id}`, which will be tested through Nginx below.

## 4. Create the systemd service

Create `/etc/systemd/system/structurerelations.service`:

```ini
[Unit]
Description=StructureRelations FastAPI web application
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=structurerelations
Group=structurerelations
WorkingDirectory=/opt/StructureRelations
Environment=PYTHONUNBUFFERED=1
Environment=SR_PLOT_CACHE_MAX_MB=32
ExecStart=/opt/StructureRelations/.venv/bin/python -m uvicorn main:app --app-dir src/webapp --host 127.0.0.1 --port 8101
Restart=on-failure
RestartSec=5
TimeoutStopSec=30

# The application writes sessions below WorkingDirectory.
ReadWritePaths=/opt/StructureRelations/webapp_sessions

[Install]
WantedBy=multi-user.target
```

Load and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now structurerelations.service
sudo systemctl status structurerelations.service
```

Inspect logs when startup fails:

```bash
sudo journalctl -u structurerelations.service -n 100 --no-pager
sudo journalctl -u structurerelations.service -f
```

Confirm that only loopback is exposed:

```bash
sudo ss -ltnp | grep ':8101'
curl --fail http://127.0.0.1:8101/
```

## 5. Configure Nginx

Create `/etc/nginx/sites-available/structurerelations` and replace the
`server_name` and certificate paths with the real public hostname and
certificates:

```nginx
server {
    listen 80;
    listen [::]:80;
    server_name structures.example.org;

    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    listen [::]:443 ssl;
    server_name structures.example.org;

    ssl_certificate     /etc/letsencrypt/live/structures.example.org/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/structures.example.org/privkey.pem;

    # Large DICOM uploads and long-running relationship calculations.
    client_max_body_size 2g;
    proxy_read_timeout 1800s;
    proxy_send_timeout 1800s;
    proxy_connect_timeout 5s;

    location /ws/ {
        proxy_pass http://127.0.0.1:8101;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }

    location / {
        proxy_pass http://127.0.0.1:8101;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 1800s;
        proxy_send_timeout 1800s;
    }
}
```

Enable the site and validate the complete Nginx configuration:

```bash
sudo ln -s /etc/nginx/sites-available/structurerelations \
    /etc/nginx/sites-enabled/structurerelations
sudo nginx -t
sudo systemctl reload nginx
```

If the default Nginx site is not needed, disable it so its catch-all behavior
does not obscure hostname mistakes:

```bash
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx
```

For a new public hostname, obtain the certificate using the server's approved
ACME process, or install certificates supplied by the organization's PKI.
Do not expose the backend port directly to the network.

## 6. Firewall and operational checks

Allow only the public web ports. Keep port 8101 restricted to loopback:

```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
sudo ufw status verbose
```

Run the following checks from a client that can resolve the public hostname:

```bash
curl --fail -I https://structures.example.org/
curl --fail https://structures.example.org/
sudo systemctl is-active structurerelations.service
sudo ss -ltnp | grep -E ':(80|443|8101)'
```

Then upload a small DICOM RT Structure Set through the UI and verify that:

1. The upload completes without a `413` or timeout.
2. Processing progress continues through the WebSocket.
3. The relationship matrix and exports load.
4. A restart preserves sessions as expected:

   ```bash
   sudo systemctl restart structurerelations.service
   ls -lh /opt/StructureRelations/webapp_sessions
   ```

## Updates and rollback

Keep the application directory and service account consistent during updates:

```bash
sudo systemctl stop structurerelations.service
cd /opt/StructureRelations
sudo -u structurerelations git pull --ff-only
sed '/^python==/d' src/webapp/requirements.txt > /tmp/structurerelations-requirements.txt
sudo -u structurerelations .venv/bin/pip install \
    -r /tmp/structurerelations-requirements.txt
sudo systemctl start structurerelations.service
sudo systemctl status structurerelations.service
```

If a deployment fails, restore the previous application revision, reinstall its
requirements, and inspect the service journal before restarting Nginx.

## Security notes

- Use a dedicated Unix account and do not run Uvicorn as root.
- Bind Uvicorn to `127.0.0.1`, not `0.0.0.0`.
- Restrict write access to the application and session directories.
- Keep TLS termination and public access in Nginx.
- Pickle-backed sessions must only be exposed to trusted users and trusted
  application input; do not treat this deployment as a public anonymous service.
- Configure log rotation and monitor disk usage for `webapp_sessions`.
