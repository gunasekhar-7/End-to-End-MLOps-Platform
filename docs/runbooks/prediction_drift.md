# Runbook: Prediction Drift Detected

## Alert
**FraudPredictionDrift** - Fraud prediction distribution drifted > 2σ from baseline for 10 minutes

## Impact
- Model predictions may be unreliable
- Business decisions based on predictions at risk
- Potential fraud losses or false positives

## Diagnosis

### 1. Check Drift Dashboard
```bash
# Grafana: Fraud Detection - Data Drift Monitoring
# Panels:
# - "Prediction Drift (Std Deviations)"
# - "Baseline vs Current Fraud Rate"
# - "Prediction Distribution"
```

### 2. Run Drift Detection Manually
```bash
# Local
make monitor

# Or directly
python -m src.monitor.drift

# Check report
cat reports/drift/drift_report.json | jq .
```

### 3. Analyze Drift Report
```json
{
  "prediction_drift": {
    "baseline_fraud_rate": 0.02,
    "current_fraud_rate": 0.08,
    "std_deviations": 3.5,
    "drift_detected": true,
    "threshold": 2.0
  }
}
```

### 4. Check Data Drift
```bash
# Check feature drift
cat reports/drift/drift_report.json | jq '.data_drift'

# Look for high PSI/KS features
```

### 5. Verify with Recent Predictions
```bash
# Check actual prediction logs
docker logs mlops-api | grep "fraud_probability" | tail -20

# Or check Prometheus
# fraud_prediction_probability{job="fraud-api"}
```

## Common Causes

| Cause | Indicators | Action |
|-------|------------|--------|
| New fraud pattern | Sudden fraud rate spike | Retrain with new data |
| Data pipeline issue | Feature drift + prediction drift | Fix upstream data |
| Seasonal change | Gradual drift, known pattern | Seasonal model / retrain |
| Model degradation | Slow drift over weeks | Scheduled retrain |
| Adversarial attack | Unusual feature values | Block/flag, investigate |
| Label delay | Drift after label arrival | Wait for labels, then retrain |

## Remediation Steps

### Immediate (15 min)
1. **Verify it's not a data issue**
   - Check upstream data source
   - Verify feature engineering pipeline

2. **Check if model needs immediate action**
   - Is fraud rate > 5x baseline? → Emergency retrain
   - Is fraud rate slightly elevated? → Schedule retrain

3. **Enable enhanced monitoring**
   ```bash
   # Increase drift check frequency
   # Edit docker-compose.yml drift-detector cron
   # Or run manually every hour
   ```

### Short-term (1 hour)
1. **Trigger retraining**
   ```bash
   # Manual trigger via GitHub Actions
   gh workflow run retrain.yml -f force_retrain=true
   
   # Or locally
   make train-full
   ```

2. **Analyze drifted features**
   ```bash
   # Get top drifted features
   cat reports/drift/drift_report.json | jq '.data_drift.drifted_features'
   
   # Check if new categories/values appeared
   ```

3. **Compare with ground truth (if available)**
   - Check delayed labels
   - Compute actual performance

### Long-term
1. **Retrain with latest data**
   - Ensure new patterns captured
   - Validate on holdout set

2. **Update drift thresholds**
   - Adjust `params.yaml` monitoring thresholds
   - Reduce false positives

3. **Add feature monitoring**
   - Track specific drifted features
   - Alert on individual feature drift

## Emergency Retrain
```bash
# 1. Pull latest data
dvc pull

# 2. Full pipeline
dvc repro

# 3. Check gates
cat metrics/train.json | jq .

# 4. If gates pass, promote
# MLflow UI → Models → fraud-detection → Transition to Staging

# 5. Deploy
# GitHub Actions CD workflow or manual docker deploy
```

## Monitoring After Fix
- Watch prediction drift panel for 24h
- Verify fraud rate returns to baseline
- Check model performance metrics
- Confirm no alert recurrence

## Escalation
- **< 1 hr**: Verify, schedule retrain
- **1-4 hr**: Emergency retrain if severe
- **> 4 hr**: Page ML team, consider model rollback

## Rollback (If New Model Worse)
```bash
# 1. In MLflow UI: Transition current Production back to Staging
# 2. Transition previous version to Production
# 3. Redeploy API with previous model
docker compose -f docker/docker-compose.yml up -d api
```

## Prevention
- Weekly automated retrain
- Feature monitoring dashboards
- Data quality checks in pipeline
- Adversarial detection
- Label feedback loop (when available)

## Related Runbooks
- [Model Stale](model_stale.md)
- [High Latency](high_latency.md)
- [Data Drift](data_drift.md)