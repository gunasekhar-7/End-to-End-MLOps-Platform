# Runbook: Model Stale (No Retrain in 30+ Days)

## Alert
**FraudModelStale** - Model not retrained in 30+ days

## Impact
- Model performance degradation over time
- Concept drift accumulation
- Regulatory/compliance risk

## Diagnosis

### 1. Check Model Age
```bash
# Grafana: Data Drift dashboard
# Panel: "Model Age (Days)"

# PromQL:
(time() - fraud_model_last_trained_timestamp) / 86400
```

### 2. Check MLflow Model Registry
```bash
# MLflow UI: http://localhost:5000
# Models → fraud-detection → Versions
# Check "Last Updated" for Production version
```

### 3. Check Recent Training Runs
```bash
# MLflow UI: Experiments → fraud-detection
# Sort by "Start Time" descending

# Or CLI:
mlflow runs list --experiment-name fraud-detection --order-by "start_time DESC" --max-results 10
```

### 4. Check Retrain Workflow
```bash
# GitHub Actions: Retrain workflow runs
# Check last successful run

# Or check logs:
gh run list --workflow=retrain.yml --limit 5
```

## Common Causes

| Cause | Fix |
|-------|-----|
| Scheduled workflow disabled | Enable in GitHub Actions |
| Drift detection not running | Check drift-detector container |
| No new data | Verify data pipeline |
| Training failures | Fix training pipeline |
| Manual oversight | Automate + alert |

## Remediation Steps

### Immediate (30 min)
1. **Verify model actually needs retrain**
   - Check drift dashboard - any drift detected?
   - Check performance metrics - degrading?

2. **Trigger manual retrain**
   ```bash
   # Via GitHub Actions
   gh workflow run retrain.yml -f force_retrain=true
   
   # Or locally
   make train-full
   ```

3. **If retrain succeeds**
   - Verify gates pass
   - Promote to Staging
   - Deploy to production

### Short-term (2 hours)
1. **Fix automation**
   - Check GitHub Actions schedule
   - Verify drift-detector cron
   - Fix any failed workflows

2. **Investigate root cause**
   ```bash
   # Check drift-detector logs
   docker logs mlops-drift-detector
   
   # Check cron
   docker exec mlops-drift-detector crontab -l
   ```

3. **Update alerting**
   - Reduce staleness threshold if needed
   - Add warning at 14 days

### Long-term
1. **Improve automation reliability**
   - Add workflow failure alerts
   - Add retrain success/failure notifications
   - Implement dead man's switch

2. **Add model freshness to CI**
   - Fail CI if model > 30 days old
   - Require retrain before deploy

3. **Document retrain schedule**
   - Calendar reminders
   - Runbook for manual retrain

## Emergency: Model Critically Stale (> 60 days)
```bash
# 1. Immediate retrain
make train-full

# 2. If no new data available
#    - Use existing data with different seed
#    - Or collect emergency data sample

# 3. Deploy with extra validation
#    - Extended test set evaluation
#    - Shadow mode deployment
#    - Gradual traffic shift
```

## Verification After Retrain
- [ ] New model in MLflow Registry (Staging)
- [ ] Gates passed (check `metrics/train.json`)
- [ ] Docker images built with new tag
- [ ] Deployed to staging/production
- [ ] Drift detector sees fresh model
- [ ] Alert resolves

## Prevention
- Weekly scheduled retrain (Monday 2 AM)
- Drift-triggered retrain
- Model age dashboard (visible to team)
- Slack notification 7 days before stale
- CI gate: fail if model > 30 days

## Related Runbooks
- [Prediction Drift](prediction_drift.md)
- [Data Drift](data_drift.md)
- [Retrain Failure](retrain_failure.md)