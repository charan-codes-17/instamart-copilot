### B1. RFM table
Snapshot 2025-10-01 00:00:00, completed orders only. 1,175 customers. Recency median 19 days (min 0, max 180); frequency mean 9.46 (max 33); monetary mean INR 2,552.06 (max INR 14,766.55).

### B2. K-means
RFM columns standardised with StandardScaler; `KMeans(n_clusters=K, n_init=10, random_state=42)`.

| K | Inertia | Silhouette |
|---|---|---|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

The elbow is at K=4: inertia falls by 41% from K=3 to K=4 (1268.2 to 743.1) but only 17% from K=4 to K=5. K=4 also has the highest silhouette (0.526) among {3, 4, 5} and gives four segments the business can act on, so **final K = 4**.
Inertia always falls as K grows because more centroids can always sit closer to the points (at K = number of customers it would be 0), so K=8 beats K=2 automatically. That fact alone cannot pick K; we need the elbow, the silhouette and interpretability.

### B3. Segments
Cluster numbers are only labels and carry no ordering.

| Segment | Name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Premium | Recommended action |
|---|---|---|---|---|---|---|---|
| 0 | Casual Regulars | 436 | 28.86 | 3.77 | 1,057.62 | No | Category-based offers to lift order frequency |
| 1 | Premium Big Spenders | 138 | 24.12 | 6.77 | 7,129.70 | Yes | Priority support and early access to electronics offers |
| 2 | Loyal Frequent Shoppers | 419 | 8.77 | 19.40 | 3,410.02 | No | Loyalty rewards; keep delivery fast |
| 3 | Lapsing Dormant | 182 | 109.04 | 2.20 | 685.99 | No | Win-back coupon campaign |

Premium = the segment with the highest mean monetary value. PCA (2 components) on the scaled RFM matrix: explained variance 0.634 and 0.218, total 0.852. Output: `outputs/customer_segments.csv`.

### B4. Lapse problem
Cutoff 2025-09-01. Cohort: customers with at least one completed order before the cutoff = 1,156. Label `lapsed = 1` if no completed order dated 2025-09-01 to 2025-09-30: 443 lapsed (38.32%). Split `train_test_split(test_size=0.2, stratify=y, random_state=42)`: 924 train, 232 test. Baseline "always not lapsed" on test: accuracy 61.6%, lapsed-class F1 0.
Features (orders before the cutoff only): recency, frequency, monetary, average discount_pct, cancellation rate, average delivery_minutes, favourite category, city, age, gender, tenure in days. No order dated on or after 2025-09-01 entered any feature, and the task 3 segment labels were not used as features.

### B5. Preprocessing without leakage
A ColumnTransformer inside a scikit-learn Pipeline: StandardScaler on numeric columns and OneHotEncoder(drop="first") on city, gender and favourite category. No encoder, scaler or resampler saw test rows: they are fitted on training rows only (and, inside GridSearchCV, on each training fold only). An assertion confirms no feature column has a missing value.
Scaler-learned mean of frequency (train only) = 8.1147 versus the full-cohort mean 8.0285; they differ because the scaler sees only the 924 training rows, not the test rows.

### B6. Tuning and evaluation
| Model | Grid | Best params | Best CV F1 |
|---|---|---|---|
| Logistic regression | C in {0.01, 0.1, 1, 10, 100} | C=100 | 0.6506 |
| Random forest | max_depth {3,5,8} x min_samples_leaf {5,10,20} | depth 8, leaf 10 | 0.6811 |

Test set (lapsed class as positive; confusion matrix rows = actual, columns = predicted [not lapsed, lapsed]):

| Model | Confusion matrix | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline (never lapsed) | [[143, 0], [89, 0]] | 0 | 0 | 0 |
| Logistic regression | [[111, 32], [31, 58]] | 0.644 | 0.652 | 0.648 |
| Random forest | [[113, 30], [31, 58]] | 0.659 | 0.652 | 0.655 |

**Selected model: random forest** (chosen on cross-validated F1, test F1 0.655 is at least 0.55). Missing a customer who is about to lapse costs all of that customer's future revenue, while wrongly targeting a loyal customer costs only a small coupon or message, so recall matters more than precision here. Both models reach the same recall (0.652), but the forest has the higher cross-validated F1 (0.681 versus 0.651) and slightly higher test precision (0.659 versus 0.644), so it wastes fewer offers. If the team wants to catch more lapsers, the decision threshold can be lowered on top of this model.

### B7. Class-balance experiment
Lapsed training rows were randomly duplicated (with replacement, random_state=42) until both classes had 570 rows; the best logistic regression was refitted; the test set is unchanged.

| Logistic regression | Precision | Recall | F1 |
|---|---|---|---|
| Original training set | 0.644 | 0.652 | 0.648 |
| Balanced training set | 0.589 | 0.854 | 0.697 |

Recall moved most (+0.20); precision did not rise (it fell by 0.055). For the retention team this means catching about 85% of lapsing customers instead of 65%, at the price of about 41% of the flagged customers being loyal ones, which is acceptable only if the offer is cheap.

### B8. Interpretation and persistence
Top 5 random-forest importances: frequency 0.280, monetary 0.271, recency 0.115, avg_delivery 0.061, cancel_rate 0.053. How often and how much a customer buys are the strongest signals; time since last order and slow deliveries come next.
Logistic regression coefficients (standardised features, C=100): frequency -1.70 (more orders, much less likely to lapse); recency +0.04 (a longer gap makes lapse slightly more likely, a very weak effect); monetary +0.28 (positive once frequency is held fixed, which is counter-intuitive and is most likely an artefact of monetary and frequency being strongly correlated, so it should not be read as "big spenders lapse").
The selected model was saved with `joblib.dump` to `outputs/lapse_model.joblib`; the reloaded model reproduces the original test predictions exactly.
Retention list (`outputs/retention_list.csv`): top 10 Premium customers of the test set by predicted lapse probability (highest 0.487); 1 of the 10 truly lapsed. This list is illustrative and not a leakage-free evaluation, because the Premium flag comes from the whole-window RFM of task 3, which includes the September window that defines the lapse label.