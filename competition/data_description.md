# Data description (notes)

Read from the Data tab on 2026-10-16. 11 Parquet files, 198 MB, licence CC BY-NC 4.0. Identifiers (customer_id, address_id, policy_id, row_id, claim_id) are text and must be read as text.

## Files

| File | One row per | Contents |
|---|---|---|
| customers | Customer | Customer details, shared across their policies |
| addresses | Address | Location and property characteristics |
| weather | Address and year | Annual weather and climate measures |
| policies_motor | Policy period | Motor training data, 2021 to 2024 |
| policies_buildings | Policy period | Buildings training data, 2021 to 2024 |
| policies_contents | Policy period | Contents training data, 2021 to 2024 |
| claims | Claim | All reported claims on the training policies, valued at 31 December 2024 |
| test_motor, test_buildings, test_contents | Policy period | 2025 rows to predict |
| sample_submission | Test row | Submission format |

## Links and structure

- customer_id links policies to customers; address_id links policies to addresses and weather; policy_id links claims to policies.
- A policy keeps its policy_id across years, so each claim is matched to a policy row by its loss_date falling in the row's period.
- A row is one policy for one calendar year or part of one. A policy can have several rows in a year after a mid-term change (new vehicle, new address). Rows end early on cancellation or lapse. `exposure` is the row's length in years.
- One customer can hold motor, buildings and contents policies; several customers can share an address. For motor, address_id is the policyholder's home address.

## Columns

**customers:** customer_id, surname (anonymised), date_of_birth, occupation, marital_status, household_size, credit_band (A best to E), prior_claims_5y (declared, before joining), licence_points, years_at_address, customer_since_year, email_domain, marketing_consent, document_preference, previous_insurer, campaign_code, branch_id, marketing_segment.

**addresses:** address_id, region (R01 to R12), urban_density, flood_zone (1 lowest to 5), distance_to_river_km, elevation_m (above local river level), soil_type, year_built, property_type, roof_type, basement, trees_within_5m, listed_building, bedrooms, energy_rating (A to G), solar_panels.

**weather:** address_id, year, hail_days, wind_exposure_index (0 to 10), freezing_days, dry_summer_index (0 wet to 1 very dry), wildfire_index (0 to 1), rainfall_mm, mean_temperature_c.

**All policy and test files:** row_id, policy_id, customer_id, address_id, period_start, period_end, exposure, voluntary_excess (0, 250, 500 or 1,000 CU), compulsory_excess, channel (direct, comparison_site, broker), payment_method (annual, monthly), missed_payments_12m, new_business, renewal_count, days_quote_to_inception, discount_applied, written_premium (premium under the insurer's current pricing), customer_service_contacts, renewal_decision (training files only: renewed, lapsed_by_customer, cancelled, declined_by_insurer, mid_term_change).

**Motor:** driver_age, years_licensed, ncb_years, vehicle_value_group (1 to 20), vehicle_age, make_model, engine_size_cc, body_type, transmission, seats, vehicle_colour, fuel_type, driver_assist, cover_type (comprehensive, third_party_fire_theft, third_party_only), annual_mileage, overnight_parking, vehicle_use, named_young_driver, telematics_opt_in, telematics_score (1 to 100, blank if not opted in).

**Buildings:** policyholder_age, sum_insured, construction_type, occupancy (owner_occupied, let), days_unoccupied, smoke_alarms, accidental_damage_cover, mortgage.

**Contents:** policyholder_age, sum_insured, tenure (owner, renter), security_level, high_value_items, accidental_damage_cover, works_from_home, lodgers, bicycle_cover.

**claims:** claim_id, policy_id, line, peril, loss_date, report_date, status (closed, open, repudiated), paid_to_date (net of excess), incurred_to_date (paid plus case estimate for open claims), recoveries, repairer_id (motor damage), third_party_id (motor third-party).

**Perils.** Motor: own_damage, third_party_prop, bodily_injury, theft, glass_hail. Buildings: storm, flood, escape_of_water, subsidence, fire, accidental_damage. Contents: theft, escape_of_water, accidental_damage, flood.

**Valuation.** Claims are valued at 31 December 2024. Some 2024 claims were not yet reported, and open claims may still develop. The target is ultimate cost, so the training target has to be developed to ultimate.

**sample_submission:** row_id, prediction (CU, zero or above). One file covers every row of all three test files.

## Flags for intake (to check against the data, not yet verified)

- `renewal_decision` exists in training only and describes what happened at the end of the row: not known at the point of pricing.
- `written_premium` is the current pricing: a ready-made incumbent to compare against, and a possible input. Whether to use it is a judgment.
- `surname`, `email_domain`, `marketing_segment`, `campaign_code`, `branch_id` raise proxy and fairness questions for a rating plan.
- Training years 2021 to 2024 and a 2025 test: validation should hold out the latest period, and the 2024 claims are the least developed.
