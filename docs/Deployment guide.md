# StructureRelations Deployment on Windows Server 2025 with Apache (Shared Host Safe)

## Purpose
This guide describes how to deploy the StructureRelations FastAPI webapp behind Apache on Windows Server 2025 when multiple webapps are hosted on the same Apache instance, with strict isolation to prevent cross-app interference.

## Why this app needs special proxy handling
This webapp expects root-based routes and uses both HTTP and WebSocket endpoints.

Relevant implementation points:
- Static mount at main.py
- Root route at main.py
- WebSocket endpoint at main.py
- Frontend API calls using absolute /api paths at app.js
- Frontend WebSocket URL built as /ws/{sessionId} at app.js
- Session storage defaults to project-local webapp_sessions at session_manager.py

## Deployment model
1. Run Uvicorn on loopback only (127.0.0.1) on a unique backend port.
2. Reverse proxy through Apache per hostname (recommended), not path prefix.
3. Keep one dedicated Windows service for this app.
4. Separate logs, temp, and storage per app.

## Recommended backend port mapping (example)
- StructureRelations: 127.0.0.1:8101
- App A: 127.0.0.1:8102
- App B: 127.0.0.1:8103

## Apache module requirements
Enable:
- mod_proxy
- mod_proxy_http
- mod_proxy_wstunnel
- mod_headers
- mod_ssl
- mod_reqtimeout

## Main Apache include order (deterministic and safe)
In the main Apache config, include vhosts in fixed order:
1. 000-default-deny.conf
2. 100-app-a.conf
3. 110-app-b.conf
4. 120-structurerelations.conf

Important rules:
- Do not define global ProxyPass rules outside VirtualHost blocks.
- Keep the default catch-all vhost first.
- Keep each app bound to a unique ServerName.

## Default catch-all vhost (safety net)
Use a default vhost so unknown host headers do not land in a real app.

    <VirtualHost *:80>
        ServerName default.invalid
        Redirect 404 /
        ErrorLog  "logs/default_http_error.log"
        CustomLog "logs/default_http_access.log" combined
    </VirtualHost>

    <VirtualHost *:443>
        ServerName default.invalid

        SSLEngine on
        SSLCertificateFile "C:/Apache24/conf/ssl/default.crt"
        SSLCertificateKeyFile "C:/Apache24/conf/ssl/default.key"

        <Location />
            Require all denied
        </Location>

        ErrorLog  "logs/default_https_error.log"
        CustomLog "logs/default_https_access.log" combined
    </VirtualHost>

## Example App A vhost
    <VirtualHost *:80>
        ServerName app-a.example.com
        Redirect permanent / https://app-a.example.com/
    </VirtualHost>

    <VirtualHost *:443>
        ServerName app-a.example.com

        SSLEngine on
        SSLCertificateFile "C:/Apache24/conf/ssl/app-a.crt"
        SSLCertificateKeyFile "C:/Apache24/conf/ssl/app-a.key"

        ProxyPreserveHost On
        RequestHeader set X-Forwarded-Proto "https"

        ProxyPass        "/" "http://127.0.0.1:8102/"
        ProxyPassReverse "/" "http://127.0.0.1:8102/"

        ErrorLog  "logs/app-a_error.log"
        CustomLog "logs/app-a_access.log" combined
    </VirtualHost>

## Example App B vhost
    <VirtualHost *:80>
        ServerName app-b.example.com
        Redirect permanent / https://app-b.example.com/
    </VirtualHost>

    <VirtualHost *:443>
        ServerName app-b.example.com

        SSLEngine on
        SSLCertificateFile "C:/Apache24/conf/ssl/app-b.crt"
        SSLCertificateKeyFile "C:/Apache24/conf/ssl/app-b.key"

        ProxyPreserveHost On
        RequestHeader set X-Forwarded-Proto "https"

        ProxyPass        "/" "http://127.0.0.1:8103/"
        ProxyPassReverse "/" "http://127.0.0.1:8103/"

        ErrorLog  "logs/app-b_error.log"
        CustomLog "logs/app-b_access.log" combined
    </VirtualHost>

## StructureRelations vhost with per-vhost long-job timeout tuning
    <VirtualHost *:80>
        ServerName structurerelations.example.com
        Redirect permanent / https://structurerelations.example.com/
    </VirtualHost>

    <VirtualHost *:443>
        ServerName structurerelations.example.com

        SSLEngine on
        SSLCertificateFile "C:/Apache24/conf/ssl/structurerelations.crt"
        SSLCertificateKeyFile "C:/Apache24/conf/ssl/structurerelations.key"

        ProxyPreserveHost On
        RequestHeader set X-Forwarded-Proto "https"

        # Per-vhost timeout tuning for long DICOM jobs
        Timeout 1800
        KeepAlive On
        KeepAliveTimeout 30
        MaxKeepAliveRequests 1000
        RequestReadTimeout header=20-40,MinRate=500 body=30-60,MinRate=500

        # Optional upload cap (example 2 GB)
        # LimitRequestBody 2147483648

        # WebSocket route first
        ProxyPass        "/ws/" "ws://127.0.0.1:8101/ws/" connectiontimeout=5 timeout=3600 keepalive=On
        ProxyPassReverse "/ws/" "ws://127.0.0.1:8101/ws/"

        # HTTP routes
        ProxyPass        "/" "http://127.0.0.1:8101/" connectiontimeout=5 timeout=1800 keepalive=On retry=0
        ProxyPassReverse "/" "http://127.0.0.1:8101/"

        ErrorLog  "logs/structurerelations_error.log"
        CustomLog "logs/structurerelations_access.log" combined
    </VirtualHost>

Notes:
- Put /ws before / to avoid route shadowing.
- Keep these settings in this vhost only so other apps keep their own timeout behavior.

## Uvicorn service isolation (Windows)
Run this app as its own service with unique runtime settings.

Example NSSM setup:
1. Install service:
   nssm install StructureRelations-Uvicorn python.exe -m uvicorn main:app --host 127.0.0.1 --port 8101 --app-dir src\webapp
2. Set working directory to project root.
3. Set dedicated stdout and stderr logs.
4. Set service-specific TEMP and TMP directories.
5. Start service.

Recommended service properties:
- AppDirectory: project root
- AppStdout: project-local log file
- AppStderr: project-local log file
- TEMP and TMP: project-local temp folder

## Security and interference prevention checklist
1. Unique ServerName per app.
2. Unique backend port per app.
3. No global ProxyPass rules.
4. Per-vhost logs (access and error).
5. Uvicorn bound to 127.0.0.1 only.
6. Dedicated Windows service per app.
7. Dedicated temp and log locations per app.
8. Default catch-all vhost denies unknown hosts.

## Validation steps
1. Apache config test passes.
2. Apache restart succeeds.
3. Each hostname serves only its corresponding backend.
4. StructureRelations upload and long processing succeed.
5. WebSocket status in UI remains connected during long jobs.
6. Logs appear only in this app’s log files.

## Operational tuning guidance
If large uploads fail:
1. Increase RequestReadTimeout body window.
2. Increase LimitRequestBody.
3. Verify reverse-proxy timeout values are not lower than expected processing duration.

If long processing jobs fail:
1. Increase ProxyPass timeout on / from 1800 upward.
2. Keep /ws timeout greater than HTTP timeout for status updates.
3. Verify backend service does not restart under load.
