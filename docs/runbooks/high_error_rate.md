# Runbook: High Error Rate

## Alert
**FraudAPIHighErrorRate** - 5xx error rate > 1% for 2 minutes

## Impact
- Failed predictions for clients
- Data loss risk
- SLA breach

## Diagnosis

### 1. Check Error Breakdown
```bash
# Grafana: API Health dashboard
# Panel: "Error Rate"

# PromQL - 5xx by endpoint:
rate(http_requests_total{job="fraud-api",status=~"5.."}[5m]) by (handler)

# PromQL - 4xx by endpoint:
rate(http_requests_total{job="fraud-api",status=~"4.."}[5m]) by (handler)
```

### 2. Check Application Logs
```bash
# Recent errors
docker logs mlops-api --tail 100 | grep -i error

# Follow logs
docker logs mlops-api -f
```

### 3. Check Specific Endpoints
```bash
# Health endpoint
curl -v http://localhost:8000/health

# Predict endpoint
curl -v -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"amount": 100}'
```

## Common Causes & Fixes

| Cause | Error Type | Fix |
|-------|------------|-----|
| Model not loaded | 503 Service Unavailable | Check startup logs, restart |
| Invalid input | 422 Validation Error | Fix client, add defaults |
| Preprocessing fail | 500 Internal Error | Check feature compatibility |
| OOM Kill | Container restart | Increase memory limit |
| Dependency down | 503/504 | Check MLflow/MinIO health |
| Timeout | 504 Gateway Timeout | Increase timeout, optimize |

## Remediation Steps

### Immediate (5 min)
1. **Check container status**
   ```bash
   docker ps -a | grep mlops-api
   # Look for "Restarting" or "Exited"
   ```

2. **Restart if crashed**
   ```bash
   docker compose -f docker/docker-compose.yml restart api
   ```

3. **Check recent deploy**
   ```bash
   # If just deployed, rollback
   git log --oneline -5
   ```

### Short-term (15 min)
1. **Analyze error patterns**
   ```bash
   # Group by error type
   docker logs mlops-api --since 30m | grep ERROR | awk '{print $NF}' | sort | uniq -c | sort -rn
   ```

2. **Fix input validation**
   - Add missing feature defaults in `serve/main.py`
   - Update Pydantic models

3. **Check model compatibility**
   ```bash
   # Verify model loads
   docker exec mlops-api python -c "
   import joblib
   model = joblib.load('models/model.pkl')
   preproc = joblib.load('models/preprocessor.pkl')
   print('Model loaded OK')
   print(f'Features: {preproc.n_features_in_}')
   "
   ```

### Long-term
1. **Add circuit breaker** - Fail fast on repeated errors
2. **Improve validation** - Better error messages
3. **Add retries** - For transient failures
4. **Dead letter queue** - Capture failed predictions

## Escalation
- **< 5 min**: Restart, check logs
- **5-15 min**: Root cause analysis
- **> 15 min**: Page on-call, consider rollback

## Rollback
```bash
# Quick rollback to previous image
docker tag ghcr.io/org/fraud-api:sha-<previous> ghcr.io/org/fraud-api:latest
docker compose -f docker/docker-compose.yml up -d api
```

## Prevention
- Contract tests for API
- Input validation at edge
- Graceful degradation
- Comprehensive error logging
- Canary deployments

## Related Runbooks
- [High Latency](high_latency.md)
- [API Down](api_down.md)
- [Prediction Drift](prediction_drift.md)