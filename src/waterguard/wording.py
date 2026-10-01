"""Approved scientific wording — the single source used by the Streamlit UI, the API
(/model-info, /benchmark) and docs/FRONTEND_SPEC.md. Do not paraphrase these into
stronger claims. Numbers here are the verified benchmark results."""

WORDING = {
    "product_name": "WaterGuard AI Saudi",
    "positioning": "ML water-loss decision-support research prototype.",
    "one_liner": ("Flags severe water-loss periods from hydraulic SCADA data and suggests which pressure sensors "
                  "to inspect first. Evaluated on the BattLeDIM L-Town benchmark."),
    "benchmark_disclaimer": ("WaterGuard is motivated by water loss in Saudi Arabia but is trained and evaluated on the "
                             "BattLeDIM L-Town benchmark, a simulated international research network. None of the "
                             "measurements come from Saudi Arabia or any Saudi utility."),
    "saudi_motivation": ("Saudi Arabia's Ministry of Environment, Water and Agriculture (MEWA) identifies reducing "
                         "losses in water networks as an improvement opportunity in its National Water Strategy, which "
                         "estimates network losses at more than 25% in different regions. This is the motivation for "
                         "WaterGuard, not evidence about its performance."),
    "experimental_threshold": ("A severe water-loss period is a 5-minute step in which total benchmark leakage is at "
                               "least 40 m³/h. This is an experimental threshold chosen for this prototype (about the "
                               "80th percentile of 2018 leakage). It is not an engineering, utility, regulatory, "
                               "competition or Saudi standard."),
    "precision": ("Precision 94.2%: of the 3,643 alerts raised on the held-out test period, 3,433 occurred during "
                  "genuinely severe periods. When WaterGuard alerts, it is usually right."),
    "recall": ("Recall 38.2%: WaterGuard caught 3,433 of 8,994 severe 5-minute steps and missed 5,561. Precision of "
               "94% does not mean 94% of severe periods are detected."),
    "alert_threshold": ("An alert is raised when the risk probability is at least 0.22. This threshold was chosen on "
                        "the validation period only, never on the test period."),
    "inspection_guidance": ("Inspection guidance, not leak localisation. Pressure sensors are ranked by how far they "
                            "are below their usual level for the time of day. It suggests where to start looking; it "
                            "does not identify a leaking pipe."),
    "not_live": "Retrospective replay of the 2018 BattLeDIM benchmark. Not live monitoring.",
    "user_mode": ("Analysis of an uploaded file with the saved V2 model. No labels are needed and the model is not "
                  "retrained."),
    "different_network": ("This model was trained on the simulated L-Town network. Data from a different network "
                          "(different sensors, topology, pressures, demand patterns or operations) is not compatible "
                          "and must not be analysed with it. A different network needs its own historical data, "
                          "verified leak records, retraining, chronological validation and threshold selection."),
    "time_recency": ("The model is not tied to the year 2018: compatible L-Town data with 2026 timestamps is processed "
                     "the same way, because no calendar feature is used. Recent timestamps do not make data from "
                     "another network compatible."),
    "ground_truth": "Ground truth (benchmark leakage) is shown for evaluation only. It is never a model input.",
    "importance": ("Feature importance and explanations describe how the model uses signals. They do not show "
                   "what caused a leak."),
    "responsible_use": ("WaterGuard AI Saudi is an independent educational research prototype and is not affiliated "
                        "with or endorsed by a Saudi government entity or water utility. It has not been validated on "
                        "real Saudi network data and must not be used for operational decisions."),
}
