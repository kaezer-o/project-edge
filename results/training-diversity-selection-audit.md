# Training-only diversity selection audit

- Seeds: 2033, 2034, 2035
- Similarity threshold: 0.90
- Validation used for selection: no
- Final test evaluated: no

| Seed | Finalists | Unique train signals | Fitness-only top-K unique | Diversity-selected |
|---:|---:|---:|---:|---:|
| 2033 | 20 | 13 | 13 | 6 |

## Seed 2033

Selected ranks: [1, 6, 8, 9, 10, 13]
Rejected as redundant: [{'rank': 2, 'training_fitness': 2.0842620323124965, 'most_similar_selected_rank': 1, 'max_active_signal_jaccard': 0.975072463768116}, {'rank': 3, 'training_fitness': 2.0652816801658673, 'most_similar_selected_rank': 1, 'max_active_signal_jaccard': 0.9802784222737819}, {'rank': 4, 'training_fitness': 2.059831475794311, 'most_similar_selected_rank': 1, 'max_active_signal_jaccard': 0.975072463768116}, {'rank': 5, 'training_fitness': 2.0398626959417743, 'most_similar_selected_rank': 1, 'max_active_signal_jaccard': 0.975072463768116}, {'rank': 7, 'training_fitness': 1.855180133627589, 'most_similar_selected_rank': 1, 'max_active_signal_jaccard': 0.9773781902552204}, {'rank': 11, 'training_fitness': 1.8002729248009905, 'most_similar_selected_rank': 10, 'max_active_signal_jaccard': 1.0}, {'rank': 12, 'training_fitness': 1.7833900312580113, 'most_similar_selected_rank': 8, 'max_active_signal_jaccard': 1.0}, {'rank': 14, 'training_fitness': 1.7542593775584296, 'most_similar_selected_rank': 6, 'max_active_signal_jaccard': 0.9395894428152493}, {'rank': 15, 'training_fitness': 1.7474917164079744, 'most_similar_selected_rank': 13, 'max_active_signal_jaccard': 1.0}, {'rank': 16, 'training_fitness': 1.7049975506360013, 'most_similar_selected_rank': 6, 'max_active_signal_jaccard': 0.9406928948913682}, {'rank': 17, 'training_fitness': 1.6951140206486934, 'most_similar_selected_rank': 10, 'max_active_signal_jaccard': 0.9753086419753086}, {'rank': 18, 'training_fitness': 1.6792989378916847, 'most_similar_selected_rank': 1, 'max_active_signal_jaccard': 0.975072463768116}, {'rank': 19, 'training_fitness': 1.6391419963515508, 'most_similar_selected_rank': 8, 'max_active_signal_jaccard': 0.940026075619296}, {'rank': 20, 'training_fitness': 1.5959843437128782, 'most_similar_selected_rank': 13, 'max_active_signal_jaccard': 1.0}]

| 2034 | 20 | 12 | 12 | 6 |

## Seed 2034

Selected ranks: [1, 2, 3, 13, 16, 19]
Rejected as redundant: [{'rank': 4, 'training_fitness': 1.7171999441922483, 'most_similar_selected_rank': 3, 'max_active_signal_jaccard': 0.9774393683023125}, {'rank': 5, 'training_fitness': 1.6761430588959885, 'most_similar_selected_rank': 3, 'max_active_signal_jaccard': 0.9937958262831359}, {'rank': 6, 'training_fitness': 1.6079343392233563, 'most_similar_selected_rank': 3, 'max_active_signal_jaccard': 0.9943598420755781}, {'rank': 7, 'training_fitness': 1.5504552409156647, 'most_similar_selected_rank': 2, 'max_active_signal_jaccard': 1.0}, {'rank': 8, 'training_fitness': 1.5049596266441454, 'most_similar_selected_rank': 2, 'max_active_signal_jaccard': 0.9954467842914058}, {'rank': 9, 'training_fitness': 1.4862146166337413, 'most_similar_selected_rank': 2, 'max_active_signal_jaccard': 1.0}, {'rank': 10, 'training_fitness': 1.459374056683722, 'most_similar_selected_rank': 2, 'max_active_signal_jaccard': 1.0}, {'rank': 11, 'training_fitness': 1.4429901399368008, 'most_similar_selected_rank': 2, 'max_active_signal_jaccard': 1.0}, {'rank': 12, 'training_fitness': 1.4325702929737774, 'most_similar_selected_rank': 2, 'max_active_signal_jaccard': 1.0}, {'rank': 14, 'training_fitness': 1.376259177988324, 'most_similar_selected_rank': 13, 'max_active_signal_jaccard': 1.0}, {'rank': 15, 'training_fitness': 1.304675623470443, 'most_similar_selected_rank': 2, 'max_active_signal_jaccard': 1.0}, {'rank': 17, 'training_fitness': 1.2715985808402328, 'most_similar_selected_rank': 3, 'max_active_signal_jaccard': 0.9994359842075579}, {'rank': 18, 'training_fitness': 1.2640297088639536, 'most_similar_selected_rank': 3, 'max_active_signal_jaccard': 0.9876957494407159}, {'rank': 20, 'training_fitness': 1.082785022918951, 'most_similar_selected_rank': 2, 'max_active_signal_jaccard': 1.0}]

| 2035 | 20 | 14 | 14 | 6 |

## Seed 2035

Selected ranks: [1, 4, 5, 6, 7, 16]
Rejected as redundant: [{'rank': 2, 'training_fitness': 2.3777509527466507, 'most_similar_selected_rank': 1, 'max_active_signal_jaccard': 1.0}, {'rank': 3, 'training_fitness': 2.2182796299583782, 'most_similar_selected_rank': 1, 'max_active_signal_jaccard': 0.9938524590163934}, {'rank': 8, 'training_fitness': 1.901809934631373, 'most_similar_selected_rank': 1, 'max_active_signal_jaccard': 0.9938524590163934}, {'rank': 9, 'training_fitness': 1.8992179202902535, 'most_similar_selected_rank': 4, 'max_active_signal_jaccard': 1.0}, {'rank': 10, 'training_fitness': 1.821605527975584, 'most_similar_selected_rank': 7, 'max_active_signal_jaccard': 0.9586206896551724}, {'rank': 11, 'training_fitness': 1.7509399182084466, 'most_similar_selected_rank': 7, 'max_active_signal_jaccard': 0.9948542024013722}, {'rank': 12, 'training_fitness': 1.688872483439301, 'most_similar_selected_rank': 7, 'max_active_signal_jaccard': 0.9586206896551724}, {'rank': 13, 'training_fitness': 1.64078932025396, 'most_similar_selected_rank': 1, 'max_active_signal_jaccard': 0.9938524590163934}, {'rank': 14, 'training_fitness': 1.562249622249039, 'most_similar_selected_rank': 4, 'max_active_signal_jaccard': 1.0}, {'rank': 15, 'training_fitness': 1.4988062851630604, 'most_similar_selected_rank': 6, 'max_active_signal_jaccard': 0.9617321248741189}, {'rank': 17, 'training_fitness': 1.411399720989012, 'most_similar_selected_rank': 6, 'max_active_signal_jaccard': 0.9279538904899135}, {'rank': 18, 'training_fitness': 1.3612325364445073, 'most_similar_selected_rank': 6, 'max_active_signal_jaccard': 0.9235181644359465}, {'rank': 19, 'training_fitness': 1.3602236346528316, 'most_similar_selected_rank': 4, 'max_active_signal_jaccard': 0.9613180515759312}, {'rank': 20, 'training_fitness': 1.3065392574014811, 'most_similar_selected_rank': 4, 'max_active_signal_jaccard': 0.9755980861244019}]

## Limitations

- This is an offline finalist-selection simulation, not a change to production evolution.
- It operates only on the existing finalist pool and cannot recover candidates discarded earlier.
- Training fitness and training signals are used for selection; validation is not read by the selection algorithm.
- The validation partition remains development data and is not independent performance evidence.
- The final chronological holdout is never evaluated.
- Bid-only source data requires an estimated spread for execution-based audits; this signal-only simulation does not model fills.
