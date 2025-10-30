
# Instructions for Admin UI Layer Status Logic (for Cursor)

This document provides the logic and structure needed to implement accurate layer status detection in the Admin UI.

---

## 🧩 Purpose

Cursor should **not** attempt to infer layer status from component health alone. Instead, it should follow a clear 3-step logic to determine:

1. Whether the **Layer is deployed**
2. Whether the **Layer is running**
3. The **Component health** only if the above two are true

---

## ✅ Step-by-Step Status Logic

### 1. **Layer Deployment Registry**

Create or connect to a service that tracks whether each Layer is deployed.

- If using Kubernetes: check for Deployments or Helm chart presence
- If using Docker Compose: parse from config or `docker ps`
- If static: maintain a manifest JSON like:

```json
{
  "Layer00": true,
  "Layer01": true,
  "Layer02": false
}
```

---

### 2. **Runtime Status**

Query whether the Layer is currently running as a process/container/pod.

- Kubernetes: use `kubectl get pods` or Kubernetes API
- Docker: use `docker ps` or container inspect
- Systemd: use `systemctl is-active`

Store or expose via an API:

```json
{
  "Layer00": "running",
  "Layer01": "stopped",
  "Layer02": "starting"
}
```

---

### 3. **Component Health (Optional, if Layer is Running)**

Only check health endpoints of components if the Layer is deployed **and** running.

```json
{
  "studio-backend": "healthy",
  "admin-ui": "inactive"
}
```

---

## 🔁 Real-Time Option

To update the Admin UI dynamically:

- Use Docker or K8s event streams
- Push updates via WebSocket or Server-Sent Events
- Or fall back to polling every X seconds

---

## 🛠 Suggested Final Output Format

Expose from a single backend API like:

```http
GET /layers/status
```

```json
{
  "Layer00": {
    "exists": true,
    "running": true,
    "components": {
      "client-admin-ui": "inactive",
      "tenant-management": "active"
    }
  },
  "Layer03": {
    "exists": false
  }
}
```

---

## ✅ Summary

Avoid guessing status from health checks. Instead:

1. Confirm deployment
2. Check runtime state
3. THEN check component health

This ensures Cursor and the Admin UI remain aligned with real system state.
