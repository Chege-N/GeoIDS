---
title: 'GeoIDS: Hyperdimensional Anomaly Detection for Zero‑Day Exploits in Encrypted Traffic Using Geometric Algebra'
tags:
  - Python
  - intrusion-detection
  - geometric-algebra
  - network-security
  - anomaly-detection
  - encrypted-traffic
  - online-learning
authors:
  - name: Chege, N.
    orcid: 0009-0005-9792-5361
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 23 May 2026
bibliography: paper.bib
---

# Summary

GeoIDS is an open‑source Python library for detecting network intrusions in fully encrypted traffic without decryption.
It embeds 25 flow‑level features (packet statistics and TLS metadata) as sparse multivectors in a mixed‑signature Clifford algebra,
then computes a blade‑based anomaly score via the commutator product.
An online sliding‑window reference frame and a Generalised Pareto tail threshold enable real‑time adaptation without batch retraining.
A key innovation is *intrinsic explainability*: every alert identifies the specific feature correlation (e.g., `tls_sni_length ∧ tls_cipher_suite_id`) that triggered the detection,
giving security analysts actionable, human‑readable context.
GeoIDS ships with a Plotly Dash dashboard, Docker deployment, SIEM output (Splunk/ELK), and a Cython acceleration path.

# Statement of Need

Over 80 % of internet traffic is now encrypted with TLS 1.3 or QUIC, rendering signature‑based IDS like Snort [@snort] and Suricata [@suricata] blind to payload content.
Statistical anomaly detectors such as autoencoders [@kitsune] and Isolation Forests [@iforest] operate on flow metadata,
but they treat features as independent dimensions and miss higher‑order correlations that are characteristic of many attack patterns.
They also require periodic batch retraining as traffic evolves, and they produce opaque anomaly scores with no explanation.

GeoIDS addresses these gaps by representing network flows as elements of a Geometric (Clifford) algebra.
The outer product encodes pairwise and three‑way feature correlations in sparse multivectors, while the commutator product between a flow’s multivector and a dynamically maintained reference measures structural non‑commutativity.
This formulation naturally captures anomalies that arise from changes in feature *interactions* — not just individual feature values — and yields a score that can be decomposed into specific blade dimensions.
The online reference update (Weiszfeld algorithm in multivector space, gated by an Isolation Forest) adapts continuously without labelled data or full retraining.

GeoIDS is designed to be practical for operational environments.
It ingests PCAP, NetFlow v5/v9/IPFIX, Zeek logs, or live capture through a unified interface.
The modular architecture allows individual components (feature extractor, GA engine, detector, output writer) to be replaced or extended.
A ready‑to‑run demonstration script generates synthetic encrypted traffic with five attack categories,
including a zero‑day DoH‑based backdoor not seen during training, and achieves detection rates above 85 % at false positive rates below 1 % on this synthetic benchmark.
Full evaluation on public datasets (CIC‑IDS2017, UNSW‑NB15) is the subject of ongoing work.

Related open‑source tools include Joy [@joy], which extracts TLS metadata but does not perform anomaly detection,
and Kitsune [@kitsune], which uses autoencoders but lacks explainability and online adaptation.
GeoIDS is the first intrusion detection system to apply Geometric Algebra to encrypted traffic analysis,
offering a novel combination of explainability, online learning, and zero‑day detection capability.

# Acknowledgements

We thank the developers of NumPy, SciPy, scikit‑learn, Plotly Dash, Click, Cython, dpkt, and nfstream for their foundational libraries.

# References
