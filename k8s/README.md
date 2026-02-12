# Kubernetes Manifests

Basic Kubernetes manifests for deploying the fraud detection API.

## Files

- `deployment.yaml` — Deployment with 2 replicas, liveness/readiness probes on `/api/v1/health`, and resource limits.
- `service.yaml` — LoadBalancer service exposing port 8000.

## Usage

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```
