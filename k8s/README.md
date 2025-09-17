# Kubernetes (k8s) – AI Financial Fraud Detection

Diretório contendo manifests YAML para deploy, gerenciamento e escalabilidade da solução antifraude financeira.

## Estrutura e principais arquivos
- `deployment.yaml`: Deploy da aplicação principal no cluster
- `service.yaml`: Exposição de pods via service
- `hpa.yaml`: Autoscaling por uso de CPU/memória
- `ingress.yaml` (opcional): Roteamento externo seguro
- `configmap.yaml` e `secret.yaml`: Variáveis/configurações e segredos

## Instruções rápidas
```bash
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f hpa.yaml
```

## Boas práticas
- Não versionar segredos reais
- Separar manifests por ambiente
- Usar probes de liveness/readiness
- Integrar com Prometheus/Grafana quando possível

Consulte a documentação dos diretórios docker/ e monitoring/ para integração DevOps completa.
