| Model | Algorithm family | Key configuration | CV Macro F1 | CV Top-1 | Student-test Top-1 | Route continuity (test) | Student-test Macro F1 | Log loss (test) | Inference (ms/1000 pts) |
|---|---|---|---|---|---|---|---|---|---|
| logreg | Linear / probabilistic | C=1.0, max_iter=1000 | 0.9258 ± 0.0255 | 0.8693 | 0.8737 | 0.9317 | 0.9282 | 0.0967 | 0.52 |
| gnb | Probabilistic (Naive Bayes) | var_smoothing=1e-09 | 0.9143 ± 0.0232 | 0.8494 | 0.8490 | 0.9188 | 0.9141 | 0.8269 | 1.73 |
| knn | Instance-based | n_neighbors=25, weights=distance, sample=100,000 rows | 0.9198 ± 0.0261 | 0.8589 | 0.8686 | 0.9286 | 0.9253 | 0.1068 | 3760.17 |
| rf | Ensemble (bagging) | n_estimators=200, min_samples_leaf=5, max_features=sqrt, max_samples=0.3 | 0.9268 ± 0.0264 | 0.8710 | 0.8807 | 0.9317 | 0.9322 | 0.0709 | 114.58 |
| lgbm | Boosting | n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8 | 0.9291 ± 0.0262 | 0.8751 | 0.8830 | 0.9375 | 0.9335 | 0.0687 | 20.64 |
| mlp | Neural network | hidden_layer_sizes=[64, 32], alpha=0.0001, batch_size=2048, learning_rate_init=0.001, max_epochs=30, patience=3 | 0.9277 ± 0.0260 | 0.8725 | 0.8793 | 0.9365 | 0.9313 | 0.0729 | 2.25 |
| lgbm_tuned | Boosting | n_estimators=300, learning_rate=0.19189364171842493, num_leaves=22, min_child_samples=235, subsample=0.9741449512038068, subsample_freq=1, colsample_bytree=0.680166375390576, reg_lambda=3.6056860986820736, reg_alpha=0.007621946573524703 | 0.9291 ± 0.0277 | 0.8750 | 0.8833 | 0.9397 | 0.9336 | 0.0700 | 13.04 |
| [baseline] nearest_arbitrary | Rule (no learning) | N/A (rule) | N/A - no probabilities/classifier | 0.5540 | 0.5449 | N/A | N/A | N/A - rule gives no probabilities | N/A |
| [baseline] nearest_heading | Rule (no learning) | N/A (rule) | N/A - no probabilities/classifier | 0.8216 | 0.8295 | N/A | N/A | N/A - rule gives no probabilities | N/A |

Reference labels' own route continuity: train 0.9877, student_test 0.9850 (same definition; realistic ceiling at 1 Hz).