# UIDAI Aadhaar Analysis Project

This repository contains an end‑to‑end analysis of UIDAI Aadhaar enrolment and update activity, from raw API exports to cleaned datasets, exploratory analysis, and presentation‑ready visuals.

## Top-Level Files

- **HACKATHON_SUBMISSION.ipynb** – Main, polished notebook used for submission; loads precomputed analysis_results outputs and cleaned_data to generate all final plots, explanations, and insights.
- **aadhaar_analysis.ipynb** – Original exploratory notebook used to experiment with loading, cleaning, and aggregating the raw CSV splits.
- **aadhaar_analysis_clean.ipynb** – Refined version of the exploratory analysis with a cleaner, more streamlined pipeline.
- **run_analysis.py** – Scripted version of the core analysis pipeline that reads raw split CSVs, builds aggregated tables (national_daily, state_panel, enrolment_by_state), and writes them into analysis_results/.
- **.gitignore** – Standard ignore rules for virtual environments, raw archives, and editor/OS artefacts.

## Data Folders

- **api_data_aadhar_enrolment/**
  - Contains raw UIDAI enrolment API exports split into multiple CSV files (e.g. `api_data_aadhar_enrolment_0_500000.csv`, etc.).
  - Each row represents enrolment counts by date, state, district, pincode, and age buckets (0–5, 5–17, 18+).

- **api_data_aadhar_demographic/**
  - Raw demographic update API exports split into CSV partitions.
  - Columns track counts of demographic updates by date, state, district, pincode, and age buckets (5–17, 17+).

- **api_data_aadhar_biometric/**
  - Raw biometric update API exports split into CSV partitions.
  - Columns track counts of biometric updates by date, state, district, pincode, and age buckets (5–17, 17+).

- **cleaned_data/**
  - **enrolment_clean.csv** – Concatenated and cleaned enrolment data: standardised dates (ISO format), trimmed state/district names, zero‑padded pincodes, non‑negative counts, duplicates removed.
  - **demographic_clean.csv** – Cleaned demographic updates with the same standardisations applied to date/state/district/pincode fields.
  - **biometric_clean.csv** – Cleaned biometric updates with consistent schema and quality checks.
  - These files are the preferred inputs for advanced, cross‑dataset analysis.

- **analysis_results/**
  - **national_daily.csv** – Aggregated national‑level daily time series of enrolments and updates.
  - **state_panel.csv** – State‑level panel with total enrolments, total demographic/biometric updates, and intensity measures (e.g. updates per 1,000 enrolments).
  - **enrolment_by_state.csv** – State‑level enrolment breakdowns by age group.
  - **01_national_timeseries.html** – Interactive Plotly view of the national daily time series.
  - **02_top_states_enrolment.html** – Top states by enrolment volume and age‑group composition.
  - **03_age_wise_distribution.html** – Age‑wise enrolment distribution visualisation.
  - **04_update_intensity_scatter.html** – Scatter plot comparing demographic vs biometric update intensity by state.
  - **05_anomalies.html** – Anomaly detection view based on z‑scores over the enrolment time series.
  - **06_forecast.html** – 14‑day enrolment demand forecast visualisation.
  - **INSIGHTS_REPORT.txt** – Textual summary of key findings and storylines from the analysis.

## How to Use This Project

1. Open **HACKATHON_SUBMISSION.ipynb** to view the full analysis narrative and final visuals.
2. Use **cleaned_data/** if you want to build new models or visualisations on top of a high‑quality, standardised dataset.
3. Use **run_analysis.py** and the raw **api_data_aadhar_*/** folders if you need to fully reproduce the analysis_results/ tables from scratch.
