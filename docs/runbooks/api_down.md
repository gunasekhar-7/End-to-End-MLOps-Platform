# Runbook: Fraud Detection API Down

## Alert
**FraudAPIDown** - API unreachable for > 1 minute

## Impact
- Complete service outage
- No fraud predictions available
- Critical business impact

## Diagnosis

### 1. Verify Alert
```bash
# Check Prometheus target
# http://localhost:9090/targets
# Look for "fraud-api" - should show "DOWN"

# Direct health check
curl -v http://localhost:8000/health
# Expected: 200 OK with {"status": "healthy", "model_loaded": true}
```

### 2. Check Container Status
```bash
# Container running?
docker ps | grep mlops-api

# Container logs
docker logs mlops-api --tail 50

# Container events
docker events --filter container=mlops-api --since 10m
```

### 3. Check Docker Compose
```bash
# Service status
docker compose -f docker/docker-compose.yml ps

# Service logs
docker compose -f docker/docker-compose.yml logs api --tail 100
```

### 4. Check Dependencies
```bash
# MLflow reachable?
curl -v http://localhost:5000

# MinIO reachable?
curl -v http://localhost:9000/minio/health/live

# Network connectivity
docker exec mlops-api ping mlflow
docker exec mlops-api ping minio
```

## Common Causes & Fixes

| Cause | Symptoms | Fix |
|-------|----------|-----|
| Container crashed | `Exited (137)` OOM | Increase memory |
| Container crashed | `Exited (1)` error | Check logs, fix bug |
| Port conflict | `Bind for 0.0.0.0:8000 failed` | Check other services |
| Dependency down | MLflow/MinIO errors | Restart dependencies |
| Config error | Startup fails | Check .env, params.yaml |
| Model load fail | "Model not found" | Run training, check paths |
| Health check fail | Container restarting | Fix `/health` endpoint |

## Remediation Steps

### Immediate (2 min)
1. **Restart API container**
   ```bash
   docker compose -f docker/docker-compose.yml restart api
   ```

2. **If restart fails, check logs**
   ```bash
   docker compose -f docker/docker-compose.yml logs api --tail 100
   ```

3. **Check for OOM**
   ```bash
   docker inspect mlops-api | grep -i oom
   # Or check container exit code: 137 = OOM
   ```

### Short-term (10 min)
1. **Fix OOM**
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

2. **Fix model loading**
   ```bash
   # Verify model exists
   docker exec mlops-api ls -la models/
   
   # Re-run training if needed
   make train-full
   ```

3. **Fix dependency issues**
   ```bash
   # Restart dependencies
   docker compose -f docker/docker-compose.yml restart mlflow minio
   
   # Wait for healthy
   docker compose -f docker/docker-compose.yml ps
   ```

4. **Fix port conflicts**
   ```bash
   # Check what's on port 8000
   netstat -tulpn | grep 8000
   # Or on Windows:
   Get-NetTCPConnection -LocalPort 8000
   ```

### Recovery Verification
```bash
# 1. Health check
curl http://localhost:8000/health

# 2. Prediction test
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"amount": 100}'

# 3. Metrics endpoint
curl http://localhost:8000/metrics | head -20

# 4. Check Prometheus target
# Should show "UP" within 30s
```

## Escalation
- **< 2 min**: Restart container
- **2-10 min**: Fix root cause (OOM, config, deps)
- **> 10 min**: Page on-call, consider failover

## Failover (If Available)
```bash
# If using load balancer with multiple replicas
# 1. Scale up healthy replicas
docker compose -f docker/docker-compose.yml up -d --scale api=3

# 2. Remove unhealthy from LB
# 3. Investigate primary
```

## Post-Mortem Checklist
- [ ] Root cause identified
- [ ] Fix deployed
- [ ] Monitoring improved (earlier detection)
- [ ] Runbook updated
- [ ] Team notified

## Prevention
- Health checks in Docker Compose
- Resource limits + monitoring
- Dependency health checks
- Graceful startup (wait for deps)
- Multiple replicas + load balancer
- Circuit breakers for dependencies
- Alert on container restart count

## Related Runbooks
- [High Latency](high_latency.md)
- [High Error Rate](high_error_rate.md)
- [MLflow Down](mlflow_down.md)