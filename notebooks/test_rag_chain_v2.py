"""
Manual RAG test — see the full pipeline in action.

This script:
1. Creates a sample legal document
2. Parses and stores it in the vector DB
3. Asks questions and shows the LLM's cited answers

Run with: python notebooks/test_rag_chain.py
"""

import sys
sys.path.append(".")

from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import TextChunker
from src.retrieval.vector_store import VectorStore
from src.chains.rag_chain import RAGChain


def main():
    # Step 1: Create a sample legal document
    sample_doc = "data/sample_docs/traffic_project_knowledge.txt"
    with open(sample_doc, "w", encoding="utf-8") as f:
        f.write("""Abstract
Traffic congestion remains one of the most pressing challenges in urban transportation, leading to increased travel times, environmental pollution, and economic costs. This dissertation presents a novel approach to real-time traffic congestion management by combining Artificial Intelligence (AI) with Digital Twin technology. The system leverages OpenStreetMap (OSM) data for road networks and traffic signal locations, while utilizing the Google Maps Directions API to obtain real-time travel times. Through spatial clustering of traffic signals and a pairwise sampling strategy, congestion is quantified using a Congestion Factor (CF) metric that compares traffic-affected travel times to free-flow conditions.
A Gradient Boosting Regressor model is trained to provide short-term congestion predictions, enhancing situational awareness and enabling proactive traffic management. The integration of a SUMO-based Digital Twin allows simulation of traffic flows and rule-based signal timing optimizations in a safe, virtual environment.
Experimental evaluation within a defined Central London bounding box demonstrates that this approach achieves accurate congestion detection and prediction while maintaining efficient use of API resources and computational latency. The proposed system offers a scalable and adaptable solution for urban traffic monitoring, contributing to improved traffic management strategies and potential reductions in congestion-related impacts.
1. Introduction
1.1 Background and Motivation
Traffic congestion is more than a daily annoyance—it’s a drag on city productivity, air quality, and public health. Many urban platforms report incidents (e.g., roadworks) or origin–destination ETAs, but few offer a network-wide, segment-level view that updates fast enough to act on. Building that view usually demands expensive sensors or proprietary feeds.
This dissertation takes a practical route: use data that already exists everywhere—OpenStreetMap (OSM) for network and signal locations, and Google Maps Directions API for travel times—to estimate congestion in near real time, predict the next few minutes, and play out “what-if” signal strategies in a SUMO digital twin. The emphasis is on a solution that is low-cost, reproducible, and globally portable: change the bounding box, and the pipeline follows.
1.2 Research Problem Statement
Cities lack a scalable, data-light method that (i) detects congestion at a local, intersection-scale, (ii) predicts short-horizon conditions, and (iii) lets practitioners trial control strategies safely before deployment. Incident feeds are sparse; OD ETAs are point-to-point; probe data is costly. We need a method that fuses OSM structure with real-time ETAs, is API-efficient, and closes the loop with simulation.
1.3 Aim and Objectives
Aim: Build a bbox-driven system that detects, predicts (short horizon), and simulates congestion using OSM + Google ETAs, visualised and tested in SUMO.
Objectives:
•	O1: Ingest OSM (signals/roads) via Overpass and retrieve ETAs via Google Directions; run anywhere the two sources are available.
•	O2: Cluster nearby traffic signals and compute a Congestion Factor (CF = traffic_time / free_flow_time) per cluster.
•	O3: Deliver +5-minute predictions per cluster using a Gradient Boosting model (features: lags, time-of-day, day-of-week, trend).
•	O4: Integrate with SUMO to colour roads by CF and trial rule-based timing suggestions (simulation-only).
•	O5: Persist results in PostgreSQL with idempotent writes, UTC timestamps, and indexes for fast reads.
•	O6: Meet operational targets: <60 s per cycle; 5-minute cadence; API-efficient pairing strategy.
1.4 Research Questions
1.	How reliably can cluster-level congestion be inferred from OSM topology and Google ETAs?
2.	What pairwise sampling and pruning rules minimise API calls without losing fidelity?
3.	Do short-horizon ML models (Gradient Boosting) materially improve situational awareness over naïve baselines?
4.	How well does a SUMO digital twin reflect detected/predicted congestion when driven by these signals?
5.	Is the approach portable (bbox-configurable) and performant enough for continuous operation?
1.5 Scope and Delimitations
•	Geography: Global by design via configurable bounding box. Central London is the running example (≈117 signals → ≈35 clusters at 50 m).
•	Data: Google Directions ETAs (free-flow vs in-traffic) + OSM/Overpass (signals/roads). No TfL feeds in the final system.
•	Control: Simulation-only in SUMO (no real signal actuation).
•	Horizon: Focus on short-term (+5 min) predictions; long-term forecasting is out of scope.
•	CRS & Time: Store WGS84 coordinates; persist times as UTC and render local time at the UI if needed.
1.6 Contributions
•	Cluster-first congestion detection: Radius-based clustering around signals; compute CF per cluster for stable, localised monitoring.
•	API-efficient sampling: Unordered pair strategy with distance-based pruning; in the London case, ~169 API calls per cycle (vs >13k naïvely) with ~40 s runtime.
•	Stable identities: Deterministic cluster UIDs (hash of sorted signal IDs + radius + version) for longitudinal analysis across runs and bbox changes.
•	Short-horizon prediction: Gradient Boosting per cluster with lags/time features and a retraining policy tied to data accrual and error thresholds.
•	Operational data layer: PostgreSQL schema (clusters, cluster_signals, cluster_congestion, predicted_congestion_v2) with idempotent upserts, UTC timestamps, and targeted indexes.
•	Digital twin loop: SUMO network from OSM (netconvert), cluster-to-edge mapping, colour-by-CF rendering, and rule-based timing suggestions for safe experimentation.
•	Resilience & QA: Retry/backoff for APIs, model cache keyed by horizon, reconciliation of predictions vs actuals, and clear error states (e.g., “no data”).
1.7 Dissertation Structure
•	Section 2 reviews congestion measures, data sources, clustering, short-horizon prediction, and digital twins.
•	Section 3 details the methodology: bbox configuration, data acquisition, clustering, sampling, metrics, cadence, storage, prediction, SUMO integration, and QA.
•	Section 4 describes the implementation and execution flow.
•	Section 5 outlines the experimental setup (Central London bbox, parameters, scenarios, metrics).
•	Section 6 presents results and discussion (detection, prediction accuracy, efficiency, scalability).
•	Section 7 concludes with contributions, limitations, and future work.





2. Literature Review
2.1 Urban Traffic Congestion: Concepts and Measures
“Congestion” is fundamentally a mismatch between demand and capacity, expressed on the ground as lower speeds, unreliable journeys, and growing queues. Classic measures include speed, density, delay, queue length, and travel-time–based indices such as the Travel Time Index (TTI) and Buffer Index (reliability). These are useful but often route- or corridor-centric and can be slow to reflect localised intersection effects—precisely where drivers actually feel congestion. Recent practice leans toward time-resolved, area-based indicators that can be updated frequently and visualised at the scale of a junction or cluster of junctions. That is the niche this work targets.
2.2 Data Sources for Congestion Sensing (Incidents, OD ETAs, Probe, OSM)
Urban traffic sensing typically draws on four families of data:
•	Incident feeds (e.g., roadworks, crashes): good for situational awareness, but sparse and reactive; they miss everyday saturation without an incident.
•	Origin–Destination ETAs (e.g., Google, HERE, TomTom): strong real-time signal and globally available, but OD-pair specific; naïve use scales poorly if sampled across a whole city.
•	Probe/sensor data (loop detectors, FCD from fleets/phones): rich but often proprietary, costly, or limited by coverage bias.
•	OpenStreetMap (OSM): excellent topology and attributes (signals, classifications) at zero licence cost, but not a traffic feed—it must be fused with a live source.
The literature shows many systems combining network geometry (OSM) with dynamic travel times (OD ETAs or probes). The persistent gaps are granularity (segment or junction level), cost (API budgets), and portability (city-agnostic setups).
2.3 OpenStreetMap and Overpass in Transport Research
OSM has matured into a dependable base map for research: links, nodes, classifications (motorway/primary/secondary), and traffic-signal nodes are widely mapped. The Overpass API enables precise, bbox-scoped queries, letting researchers pull just what they need. Strengths: coverage, openness, and detail down to turn restrictions in many cities. Limitations: update latency (community mapping cycles) and heterogeneity (tagging varies by region). In practice, OSM provides the structure (where to measure and simulate), while a live source supplies the state (how traffic is moving now).
2.4 Spatial Clustering for Networked Urban Data
Clustering turns scattered observations into actionable spatial units. Two families recur in traffic analytics:
•	Radius / density-based (e.g., DBSCAN, or simple geodesic buffers): discover natural groups without pre-setting K; tolerant to irregular geometry—ideal around intersections.
•	Centroid / partition-based (e.g., K-means): efficient, but assumes convex, similarly sized clusters and can cut across junction logic.
For signal-centric monitoring, geodesic radius clustering (50 m in this work) strikes a practical balance: it captures junction-scale effects, avoids arbitrary partitions, and limits pair counts for OD sampling. A key implementation nuance from prior art is maintaining stable cluster identities across runs; without this, longitudinal analysis becomes noisy.
2.5 Short-Horizon Traffic Prediction (Classical vs ML)
Short-horizon (5–30 min) forecasting is a well-studied problem:
•	Classical models (ARIMA variants, Kalman filters): strong on stationarity and interpretability; struggle with non-linearities, regime changes, and exogenous effects without careful tuning.
•	Machine learning (tree ensembles, gradient boosting, XGBoost/LightGBM): handle non-linear, interaction-heavy patterns with modest data and compute; relatively robust to noise.
•	Deep learning (RNN/LSTM/TCN, GNNs): powerful for high-volume sensor grids but come with data hunger, complexity, and harder operationalisation.
Given the data characteristics here (cluster-level timeseries, modest volume, need for speed and robustness), tree-based boosting is an appropriate middle ground, enabling lag features and temporal cyclic encodings with fast retraining.
2.6 Digital Twins and Traffic Simulation (SUMO)
A digital twin mirrors real-world state in a simulation for safe, rapid what-if testing. In traffic, SUMO is widely used for its open tooling, OSM ingestion (netconvert), and programmatic control (TraCI). The literature shows two fruitful patterns: (1) visual alignment—colouring simulated edges by measured/predicted states for intuitive operator sense-making; (2) control prototyping—adjusting signal phases or speeds under simulated conditions before any field deployment. Challenges reported include data–model alignment and runtime at city scale; both are addressed here by focusing on cluster-level signals and bbox-scoped networks.
2.7 Synthesis and Gap Analysis
What we learn from prior work
1.	Granularity gap: Many systems report corridor or network-wide indices; fewer offer junction-scale, frequently updated views that reflect driver experience.
2.	Cost/scale gap: OD-based sensing is attractive but API-expensive if sampled naively; probe data can be locked behind paywalls.
3.	Identity gap: Clustering is common, but maintaining stable cluster identities across re-runs/bboxes is rarely addressed—hurting comparability.
4.	Prediction gap: Deep models shine with dense sensors; for lightweight, portable setups, evidence favours tree ensembles with good features, yet operational recipes (retraining, caching, reconciliation) are often under-specified.
5.	Twin-in-the-loop gap: Simulation is used for research, but closing the loop—feeding live detected/predicted states into a twin for rule testing—is less common in low-cost, open pipelines.
How this dissertation addresses the gaps
•	Junction-scale views: Use traffic signals as anchors, cluster by geodesic radius (50 m), and compute a cluster Congestion Factor (CF = traffic/free) every cycle.
•	API efficiency: Limit OD sampling to unordered pairs within clusters, with distance-based pruning—delivering ~169 calls/cycle in the London study vs >13k naïvely.
•	Stable identities: Generate deterministic cluster UIDs (hash of sorted signal IDs + radius + version) to enable longitudinal analysis and clean database keys.
•	Practical prediction: Deploy gradient boosting with lag/temporal features, horizon-keyed model caches, retraining triggers, and prediction–actual reconciliation for measurable accuracy.
•	Twin alignment: Build a SUMO network from OSM, map clusters → edges, colour by CF, and apply simulation-only, rule-based timing suggestions for safe experimentation.
In short, the literature motivates a solution that is area-aware, budget-aware, and operator-aware. This work aligns with that direction and contributes a portable, explainable, and simulation-ready approach suited to cities that lack dense sensor coverage but need reliable, near-real-time insight at the places where congestion hurts most: signalised intersections.









3. Methodology
This section explains how the system detects, predicts, and simulates congestion—what data it uses, how it transforms that data, and why each design choice was made. The guiding principles are: portable (bbox anywhere), explainable (simple metric, clear thresholds), API-efficient, and simulation-ready (SUMO).
________________________________________
3.1 System Overview and Assumptions
•	Overview. The pipeline ingests OSM (signals/roads) via Overpass, clusters nearby signals, samples Google Directions ETAs within clusters, computes a cluster Congestion Factor (CF), predicts short-horizon CF via Gradient Boosting, and pushes states into SUMO for live visualisation and rule-testing.
•	Assumptions.
o	OSM has adequate coverage of traffic signals in the chosen bbox.
o	Google Directions returns free-flow and in-traffic durations reliably for short urban hops.
o	Near-term traffic is locally autocorrelated, so short-horizon ML with lagged features is appropriate.
o	Optimisation is simulation-only (no field actuation).
________________________________________
3.2 Study Area Configuration (Bounding Boxes) and Generalisability
•	Bbox input defines the scope. Example used later (Central London):
N 51.521702, S 51.514658, E −0.116000, W −0.132447.
•	Global portability. Change the bbox; the same code runs anywhere OSM and Google have coverage.
•	CRS. Data are stored in WGS84 (EPSG:4326); distances computed geodesically.
________________________________________
3.3 Data Acquisition
3.3.1 OpenStreetMap via Overpass (Signals, Roads)
•	Query highway=traffic_signals and relevant road geometries within the bbox.
•	Extract: signal node IDs + coordinates; road classes (primary/secondary/tertiary) to contextualise clusters.
3.3.2 Google Maps Directions ETAs (Free-flow vs In-traffic)
•	For each unordered signal pair inside a cluster, request:
o	Free-flow time (baseline)
o	Time with traffic (current conditions)
•	Responses provide the raw ingredients for CF at cluster level (below). Calls are throttled and pruned to respect budget and latency targets.
________________________________________
3.4 Signal Clustering and Stable Identifiers
3.4.1 Geodesic Distance and Radius Selection
•	Radius-based clustering using geodesic distance (Haversine).
•	Default R = 50 m: tight enough to capture junction-scale behaviour; large enough to avoid micro-clusters.
3.4.2 Cluster Centroids and Membership
•	For each cluster: compute centroid (mean lat/lon) and persist member signal IDs.
•	Centroids act as anchor points for mapping and SUMO edge association.
3.4.3 Stable Cluster UID (Hashing Strategy)
•	To avoid “identity drift” across runs/bbox tweaks:
cluster_uid = SHA1("v|radius|sorted(signal_ids)")[:16]
•	Store both cluster_id (run-local) and cluster_uid (stable) so historical comparisons remain valid.
________________________________________
3.5 Pairwise Sampling Strategy
3.5.1 Unordered Pairs and Pruning Rules
•	For a cluster of size n, sample unordered pairs: (n2)\binom{n}{2}(2n).
•	Pruning. Skip pairs under a minimum geodesic distance (e.g., < 30–50 m) to avoid trivial hops and save API calls.
•	Dedup. Treat A→B and B→A as one pair; use median across routes if multiple are returned.
3.5.2 API Budget and Latency Targets
•	Target per cycle: complete all calls and processing in < 60 s (refresh every 5 min).
•	Example (Central London): ~117 signals → ~35 clusters (@ 50 m) → ∑(ni2)≈168\sum \binom{n_i}{2} \approx 168∑(2ni)≈168 Directions calls + 1 Overpass = ~169 calls/cycle, ~40 s runtime.
•	Controls: distance pruning, call batching, exponential backoff, and jittered scheduling to smooth QPS.

""")

    print("=" * 60)
    print("LEGAL DOCUMENT Q&A — RAG DEMO")
    print("=" * 60)

    # Step 2: Parse and store
    parser = DocumentParser()
    chunker = TextChunker()
    store = VectorStore()
    chain = RAGChain()

    print("\n📄 Parsing sample lease agreement...")
    documents = parser.parse(sample_doc)
    chunks = chunker.chunk(documents)
    num_stored = store.add_chunks(chunks, doc_id="sample_lease")

    print(f"   Extracted {len(documents)} page(s)")
    print(f"   Created {len(chunks)} chunks")
    print(f"   Stored {num_stored} chunks in vector DB")

    # Step 3: Ask questions
    questions = [
        "What are the apis used in the project?",
        "what is the geodesic distance and radius?",
        "What is the pruning rules for pairwise sampling strategy?",]

    for q in questions:
        print(f"\n{'─' * 60}")
        print(f"❓ Question: {q}")
        print(f"{'─' * 60}")

        result = chain.ask(q, k=3)

        print(f"\n💡 Answer:\n{result['answer']}")
        print(f"\n📚 Sources used ({result['num_sources']}):")
        for src in result["sources"]:
            print(f"   • {src['source']}, Page {src['page']} "
                  f"(relevance: {src['relevance_score']})")

    print(f"\n{'=' * 60}")
    print("Demo complete! Your RAG pipeline is working end-to-end.")
    print("=" * 60)


if __name__ == "__main__":
    main()
