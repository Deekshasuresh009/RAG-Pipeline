# Aventro Motors RAG Phase 1 - Markdown Extraction & Cleaning Report

## 1. Overview & Summary Metrics

- **Source Directory**: `Aventro Motors/md/`
- **Total Markdown Files Scanned**: 50
- **Successfully Processed Documents**: **50**
- **Empty Documents**: **0**
- **Documents with Tables Preserved**: 23
- **Documents with Lists Preserved**: 49
- **Total Raw Character Count**: 170,022 characters
- **Total Cleaned Character Count**: 169,453 characters
- **Whitespace / Redundancy Trimmed**: 569 characters
- **Output Artifact**: `artifacts/extracted/documents.jsonl`

---

## 2. Text Cleaning & Formatting Preservation

The text cleaning pipeline applies strictly non-destructive whitespace normalization:
- **Preserved**: All technical numbers, dates, currency values (e.g., Lakhs, ₹), battery and engine specs, dimensions, tables, lists, and safety procedures.
- **Cleaned**: Normalized Windows `CRLF` to `LF`, trimmed trailing line whitespace, collapsed 3+ consecutive blank lines to standard single blank lines, and converted non-breaking spaces (`\u00a0`).

---

## 3. Duplicate Content Detection

- **Unique Duplicate Passage Groups**: **2**
- **Full Document Duplicates**: **0** (all 50 documents have distinct topics and unique hashes).

### Identified Shared Passages Across Documents

#### Group 1 (Found in 3 documents)
- **Shared in**: `Aventro Glory SUV.md, Aventro Grand SUV.md, Aventro Motors Variants & Pricing.md`
- **Sample Text**: *"**Head Office:** Aventro Tower, Sector 18, Gurgaon, Haryana - 122018, India **Contact:** +91 124 1234567 **Email:** info..."*
- **Resolution**: Preserved in each document. These represent standard corporate contact footers; no documents were merged or discarded.

#### Group 2 (Found in 2 documents)
- **Shared in**: `Aventro Pulse Hatchback.md, Parts-3.md`
- **Sample Text**: *"**Aventro Motors Pvt. Ltd.** Aventro Tower, Sector 18, Gurgaon, Haryana - 122018, India Contact: +91 124 1234567 Email: ..."*
- **Resolution**: Preserved in each document. These represent standard corporate contact footers; no documents were merged or discarded.

---

## 4. Document Inventory Table

| Doc ID | Document Title | Section Headings | Tables | Lists | Cleaned Chars |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `doc_about_aventro_motors` | Aventro Motors Pvt. Ltd. | 7 | Yes | Yes | 4,088 |
| `doc_adaptive_cruise_control_acc` | Adaptive Cruise Control (ACC): What is it & How Does it Work? | 13 | Yes | Yes | 4,181 |
| `doc_adjusting_the_seat_positions` | Adjusting the Seat Positions in Aventro Motors Vehicles | 11 | Yes | Yes | 3,547 |
| `doc_adjusting_the_steering_wheel` | Adjusting the Steering Wheel | 13 | No | Yes | 2,627 |
| `doc_advice_for_pregnant_women_while_in_a_car` | Advice for Pregnant Women While in a Car | 10 | No | Yes | 3,168 |
| `doc_aventro_awards_recognitions` | Aventro Motors Pvt. Ltd. - Awards & Recognitions | 9 | No | Yes | 2,755 |
| `doc_aventro_connectivity` | Aventro Motors Guide: Connecting Smartwatch Connectivity, Works with Alexa, and CarPlay | 24 | No | Yes | 4,733 |
| `doc_aventro_aero` | Aventro Aero – Model Overview | 12 | Yes | Yes | 3,850 |
| `doc_aventro_bolt_sedan` | Aventro Motors Pvt. Ltd. | 10 | Yes | Yes | 4,502 |
| `doc_aventro_edge_sedan` | Aventro Edge Sedan | 9 | Yes | Yes | 5,286 |
| `doc_aventro_glory_suv` | Aventro Motors Pvt. Ltd. | 8 | Yes | Yes | 4,172 |
| `doc_aventro_grand_suv` | Aventro Grand SUV | 12 | Yes | Yes | 4,264 |
| `doc_aventro_motors_variants_pricing` | Aventro Motors Pvt. Ltd. | 14 | Yes | Yes | 6,476 |
| `doc_aventro_motors_how_to_change_a_tyre` | Aventro Motors: How to Change a Tyre | 17 | Yes | Yes | 3,646 |
| `doc_aventro_prime_sedan` | Aventro Prime Sedan | 10 | Yes | Yes | 4,170 |
| `doc_aventro_pulse_hatchback` | Aventro Pulse Hatchback | 11 | Yes | Yes | 3,962 |
| `doc_aventro_service_center_locations` | Aventro Motors Pvt. Ltd. – Service Center Locations | 21 | No | Yes | 3,835 |
| `doc_aventro_spark_hatchback` | Aventro Spark Hatchback | 8 | Yes | Yes | 4,848 |
| `doc_aventro_storm_suv` | Aventro Motors Pvt. Ltd. | 11 | Yes | Yes | 4,396 |
| `doc_aventro_swift_sedan` | Aventro Swift Sedan | 8 | Yes | Yes | 4,241 |
| `doc_aventrozoom_suv` | Aventro Motors Pvt. Ltd. | 12 | Yes | Yes | 3,088 |
| `doc_best_ways_to_maintain_good_mileage` | Best Ways to Maintain Good Mileage | 12 | No | Yes | 3,112 |
| `doc_best_ways_to_use_your_car_ac` | Best Ways to Use Your Car AC in All Seasons | 6 | No | Yes | 2,885 |
| `doc_best_ways_to_use_your_headlamps` | Best Ways to Use Your headlamps Based on Road, Traffic, and Climate | 12 | Yes | Yes | 4,736 |
| `doc_careers` | Careers at Aventro Motors Pvt. Ltd. | 10 | No | Yes | 3,718 |
| `doc_discounts` | Aventro Motors Pvt. Ltd. | 13 | No | Yes | 2,765 |
| `doc_do_and_donts_about_3rd_party` | Do's and Don'ts About 3rd Party Accessories & Electrical Accessories for Your Car | 18 | Yes | Yes | 4,085 |
| `doc_faq_part_1` | Frequently Asked Questions (FAQs) | 1 | No | No | 535 |
| `doc_how_to_handle_engine_wont_start_issue` | How to Handle "Engine Won't Start" Issue | 9 | No | Yes | 2,567 |
| `doc_how_to_open_the_door_when_the_car_key_remote_battery_is_dead` | How to Open the Door When the Car Key Remote Battery is Dead | 7 | No | Yes | 1,785 |
| `doc_how_to_open_the_fuel_knob_in_aventro_cars` | How to Open the Fuel Knob in Aventro Cars | 3 | No | Yes | 1,974 |
| `doc_how_to_open_the_hood_of_an_aventro_vehicle` | How to Open the Hood of an Aventro Vehicle | 4 | No | Yes | 2,114 |
| `doc_how_to_start_your_aventro_car` | How to Start Your Aventro Car | 9 | No | Yes | 2,327 |
| `doc_how_to_tow_your_aventro_car` | How to Tow Your Aventro Car | 12 | No | Yes | 2,836 |
| `doc_instrument_cluster_warning_alerts` | Instrument Cluster Warning Alerts | 23 | No | Yes | 3,608 |
| `doc_lane_keeping_assist_system` | Lane Keeping Assist System (LKAS): Overview & Functionality | 12 | No | Yes | 3,623 |
| `doc_loans` | Loan Options for Aventro Motors Vehicles | 6 | Yes | Yes | 2,930 |
| `doc_opening_and_closing_the_boot` | Opening and Closing the Boot | 11 | No | Yes | 2,335 |
| `doc_parta_1` | Aventro Motors Pvt. Ltd. | 8 | No | Yes | 2,245 |
| `doc_parts_2` | Aventro Motors Pvt. Ltd. | 8 | No | Yes | 2,551 |
| `doc_parts_3` | Aventro Motors Pvt. Ltd. | 4 | Yes | Yes | 2,237 |
| `doc_parts_4` | Aventro Motors Pvt. Ltd. | 4 | Yes | Yes | 2,363 |
| `doc_parts_5` | Aventro Motors Pvt. Ltd. | 4 | Yes | Yes | 4,629 |
| `doc_parts_6` | Aventro Motors Pvt. Ltd. | 3 | Yes | Yes | 4,059 |
| `doc_pre_owned_certified_cars` | Aventro Motors Pvt. Ltd. | 7 | No | Yes | 4,039 |
| `doc_safe_driving_checklist` | Safe Driving Checklist | 8 | No | Yes | 2,690 |
| `doc_sell_your_car_in_3_easy_steps` | Sell Your Car in 3 Easy Steps | Aventro Motors Pvt. Ltd. | 10 | No | Yes | 2,360 |
| `doc_test_drive_at_your_doorstep` | Request a Test Drive at Your Doorstep | 4 | No | Yes | 1,282 |
| `doc_traction_control` | Traction Control | 14 | No | Yes | 3,943 |
| `doc_what_is_abs` | What is ABS (Anti-lock Braking System) and How Does It Work? | 9 | No | Yes | 3,285 |
