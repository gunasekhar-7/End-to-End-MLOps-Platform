# Runbook: High API Latency

## Alert
**FraudAPIHighLatency** - p99 latency > 500ms for 5 minutes

## Impact
- Degraded user experience
- Potential timeout errors for clients
- SLA breach risk

## Diagnosis

### 1. Check Current Latency
```bash
# Grafana: Fraud Detection - API Health dashboard
# Panel: "Latency (p50, p95, p99)"

# PromQL query:
histogram_quantile(0.99, rate(http_request_duration_seconds_bucket{job="fraud-api"}[5m]))
```

### 2. Check Request Volume
```bash
# PromQL:
rate(http_requests_total{job="fraud-api"}[5m])
```
- Sudden spike? → Traffic surge
- Normal volume? → Performance degradation

### 3. Check Error Rate
```bash
# PromQL:
rate(http_requests_total{job="fraud-api",status=~"5.."}[5m])
```
- High 5xx? → Upstream dependency failure

### 4. Check Container Resources
```bash
# Grafana: System Monitoring dashboard
# Panels: CPU Usage, Memory Usage

# Docker:
docker stats mlops-api
```

### 5. Check Model Inference Time
```bash
# Check if model loading is slow
docker logs mlops-api | grep -i "load\|predict"

# Profile endpoint timing
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"amount": 100}'
```

## Common Causes & Fixes

| Cause | Symptom | Fix |
|-------|---------|-----|
| Cold start | High latency after deploy | Pre-warm: `curl /health` |
| Model too large | High memory, slow predict | Optimize model, quantize |
| CPU saturation | High CPU%, high latency | Scale horizontally |
| Memory pressure | High memory%, OOM kills | Increase memory limit |
| GC pauses | Periodic latency spikes | Tune Python GC |
| Dependency slow | MLflow/MinIO calls slow | Check network, cache |
| Batch size | Large batches slow | Limit batch size |

## Remediation Steps

### Quick Fixes (Try First)
1. **Restart API container**
   ```bash
   docker compose -f docker/docker-compose.yml restart api
   ```

2. **Scale horizontally**
   ```bash
   docker compose -f docker/docker-compose.yml up -d --scale api=3
   # Requires load balancer
   ```

3. **Increase resources**
   ```yaml
   # docker-compose.yml
   api:
     deploy:
       resources:
         limits:
           memory: 4G
         reservations:
           memory: 2G
   ```

### Investigation
1. **Profile the application**
   ```bash
   # Add py-spy to Dockerfile.serve
   # Run: py-spy top --pid <pid>
   ```

2. **Check model inference**
   ```python
   # In container:
   python -c "
   import joblib, time, numpy as np
   model = joblib.load('models/model.pkl')
   X = np.random.rand(1, 50)
   start = time.time()
   for _ in range(100):
       model.predict_proba(X)
   print(f'Avg: {(time.time()-start)/100*1000:.2f}ms')
   "
   ```

3. **Check preprocessor**
   ```python
   # Same as above but with preprocessor.transform()
   ```

### Long-term Fixes
1. **Model optimization**
   - ONNX conversion
   - Quantization (INT8)
   - Tree pruning
   - Distillation

2. **Caching**
   - Cache preprocessor output for repeated inputs
   - Redis for frequent predictions

3. **Async processing**
   - Move heavy preprocessing to background
   - Return cached results for identical requests

## Escalation
- **< 15 min**: Try quick fixes
- **15-30 min**: Investigate root cause
- **> 30 min**: Page on-call, consider rollback

## Rollback Procedure
```bash
# 1. Get previous image tag
docker images fraud-api --format "table {{.Tag}}\t{{.CreatedAt}}"

# 2. Deploy previous version
docker compose -f docker/docker-compose.yml up -d \
  --build \
  --scale api=2 \
  api

# 3. Verify latency recovered
# 4. Update DNS/load balancer
```

## Prevention
- Load test before deployments
- Set resource limits/requests
- Monitor model size growth
- Regular performance profiling
- Chaos engineering (latency injection)

## Related Runbooks
- [High Error Rate](high_error_rate.md)
- [API Down](api_down.md)
- [Model Stale](model_stale.md)