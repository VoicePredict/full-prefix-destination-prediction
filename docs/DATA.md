# Dataset scope and preprocessing

## Why GeoLife is used

The evaluated construct requires persistent identities, chronological
personal histories, complete coordinate sequences that can be truncated at
several observation ratios, and repeated destinations within a user. GeoLife
provides these properties and is large enough to keep configuration selection
and final evaluation in disjoint user cohorts. This supports construct
validity for personalized prefix-based endpoint prediction; using one corpus
still limits external validity.

Taxi, check-in, and aggregate mobility corpora are not treated as equivalent
replications. Some taxi datasets preserve vehicle or driver identifiers, but
their endpoints reflect passenger demand and service operation. Check-in
histories lack continuous prefix geometry, and aggregate data lack persistent
individual histories. Substituting one of these sources would change either
the target construct or the information supplied to the methods.

## Source unit

One original GeoLife PLT file is one trajectory throughout the pipeline. No
manual, gap-based, or label-based journey segmentation is performed. The
target is the final recorded coordinate of that source file, not a verified
semantic activity destination.

After invalid coordinate or timestamp rows are removed, a file is eligible if
it has at least 10 valid observations and spans at least five minutes.
Sequences longer than 200 points are reduced by selecting original samples at
evenly spaced index positions. The endpoints are preserved, nearest-integer
ties use ties-to-even rounding, and duplicated selected indices are removed.

## Transportation modes

GeoLife labels are time intervals rather than file labels. Removing labelled
rows would change source-file boundaries, so a complete PLT file is excluded
if it overlaps an explicitly non-ground label: airplane, boat, ferry, ship,
or helicopter. Nineteen files are removed by this rule. Ground-labelled and
unlabelled files are retained.

The 5,845 evaluation trajectories have the following dominant file-level
labels: bus 609, walk 515, bicycle 507, subway 105, car 75, taxi 25, and train
11; 3,998 files are unlabelled at the dominant-mode level. The mean labelled
point share is 28.7%, and the median is zero. Mode-specific accuracy is
therefore not estimated.

## Cohorts and chronology

Users require at least 40 eligible trajectories, leaving 76 users. The
discovery cohort contains 30 users sampled across trajectory-count tertiles.
Nine users occur in earlier preliminary analyses and are excluded from final
evaluation. The remaining 37 users form the evaluation cohort.

Within each user, trajectories are sorted by source start time. The earliest
75% form the outer history and the latest 25% form the test period. The final
20% of the outer history is validation in discovery. Once configurations are
frozen, fit and validation history are combined for personal training in the
evaluation cohort. The resulting evaluation counts are 3,497 fit, 873
validation, and 1,475 test trajectories.

The term *held-out user* refers to exclusion from configuration and threshold
selection. It does not mean zero-shot evaluation: the earlier history of each
evaluation user is required to construct that user's destination catalogue
and personal candidates.

## Destination supports and unsupported endpoints

Personal destination regions are formed from training endpoints only using
DBSCAN connectivity of 200 m and `min_samples=1`. The medoid is an observed
training endpoint. R80, R90, and R95 are training-only within-region distance
quantiles; singleton supports consequently have radius zero. D200 is retained
as a common nonzero sensitivity metric.

The implementation uses the original DBSCAN components to construct region
medoids, support counts, and empirical radii. After those quantities are
frozen, each training candidate and classifier target receives the label of
its nearest personal medoid. The two partitions differ for 66 of 4,370
evaluation-history endpoints (1.51%), across 12 users and 44 catalogue
regions. The exact audit is retained in
`reference/results/evaluation37/final/catalogue_assignment_audit.json`. A
post-hoc matched-grid sensitivity replaces the downstream labels by original
DBSCAN membership while retaining the catalogue, scoring rule, candidates,
and saved bootstrap plan.

An evaluation endpoint outside every training R90 support remains in the
denominator and is counted as an error. The public case files preserve
coverage and all-case hit indicators while omitting the underlying endpoint
coordinates.
