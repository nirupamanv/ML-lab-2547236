# Ethics Statement — AQI Risk Classification (Mission Earth)

*(This is a standalone copy of the ethics discussion in Section Q4 of the notebook, provided
separately because the CIA-3 brief lists "ethics statement" as its own required submission item.)*

## Bias / Fairness
Monitoring stations in the CPCB dataset are concentrated in large, wealthier metros (Delhi, Mumbai,
Bengaluru, etc.). Smaller cities (Aizawl, Shillong, Jorapokhar) contribute far fewer city-days. A model
trained predominantly on megacity pollution profiles may not transfer reliably to towns whose
dominant pollution sources (e.g. biomass cooking, small-scale industry) differ from metro
traffic/industrial emissions. Any deployment to a new city should be locally validated before its
predictions are trusted, not assumed to generalise automatically.

## Privacy
The dataset is exclusively environmental sensor data — pollutant concentrations, city, and date —
with no personally identifiable information. Individual-privacy risk is low. The main responsible-use
concern is aggregate misuse, e.g. a city administration selectively publicising only favourable
predictions to the public.

## Uncertainty
The model outputs class probabilities (`predict_proba`), not certainties. Deployed interfaces should
surface these probabilities (e.g. "72% Severe, 25% Very Poor") rather than collapsing every prediction
to a single hard label, particularly for readings that sit close to a category boundary.

## Cost of False Positives vs. False Negatives
A **false negative on the "Severe" class** — predicting a lower risk category when true air quality is
actually severe — is the most dangerous failure mode, since it could lead schools, hospitals, or
outdoor-worker organisations to skip necessary precautions on a genuinely hazardous day. A **false
positive** (over-warning on a day that wasn't actually severe) carries a real but much smaller cost:
public inconvenience and warning fatigue if it recurs too often. This asymmetry is why the training
pipeline uses `class_weight='balanced'` (Logistic Regression, Random Forest) and inverse-frequency
`sample_weight` (XGBoost), and why recall on high-risk classes — not just overall accuracy — was
tracked throughout model comparison.

## Human Oversight
This model should be positioned as a **decision-support tool that flags likely high-risk days for a
human analyst to confirm**, not as an autonomous public-alert system. This is especially important
given (a) the metro-city bias described above, and (b) the fact that the training data ends in 2019,
predating several years of policy, vehicle-fleet, and industrial changes.

## Limits on Deployment
1. Do not deploy to a city or region with no representation in the training data without local
   validation first.
2. Retrain periodically — pollution sources, vehicle fleets, and regulations shift over multi-year
   horizons, and a model trained on 2015–2019 data will drift.
3. Sensor calibration drift over time is not modelled here and should be monitored operationally if
   this is ever put into production.
4. This is a **same-day nowcast** from same-day pollutant readings, not a multi-day-ahead forecast —
   it should not be marketed or relied upon as predicting tomorrow's AQI.
