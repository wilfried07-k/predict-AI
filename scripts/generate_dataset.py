"""Script to generate synthetic poultry dataset."""

import csv
import random
from pathlib import Path

random.seed(42)

out_path = Path('data') / 'synthetic_poultry_farm.csv'
out_path.parent.mkdir(parents=True, exist_ok=True)

n = 50000

with out_path.open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        'age_days',
        'chick_weight_g',
        'temp_c',
        'humidity_pct',
        'stocking_density',
        'vaccine_score',
        'breed_index',
        'housing_quality',
        'feed_protein_pct',
        'water_quality',
        'management_index',
        'flock_size',
        'feed_price_usd_kg',
        'sale_price_usd_kg',
        'energy_cost_usd',
        'final_weight_kg',
        'mortality_rate_pct',
        'avg_daily_gain_g',
        'feed_intake_kg',
        'fcr',
        'annual_revenue_usd'
    ])

    for _ in range(n):
        age_days = random.randint(30, 60)
        chick_weight_g = random.uniform(35, 50)
        temp_c = random.uniform(18, 32)
        humidity_pct = random.uniform(60, 90)
        stocking_density = random.uniform(8, 14)
        vaccine_score = random.uniform(0.7, 1.0)
        breed_index = random.uniform(0.8, 1.2)
        housing_quality = random.uniform(0.7, 1.0)
        feed_protein_pct = random.uniform(18, 24)
        water_quality = random.uniform(0.7, 1.0)
        management_index = random.uniform(0.7, 1.0)
        flock_size = random.randint(500, 5000)
        feed_price_usd_kg = random.uniform(0.35, 0.75)
        sale_price_usd_kg = random.uniform(1.2, 2.5)
        energy_cost_usd = random.uniform(200, 1200)

        temp_penalty = 1.0 - min(abs(temp_c - 24) / 20, 0.6)
        hum_penalty = 1.0 - min(abs(humidity_pct - 65) / 50, 0.5)
        density_penalty = 1.0 - min((stocking_density - 10) / 10, 0.3)

        growth_factor = (
            breed_index
            * housing_quality
            * vaccine_score
            * management_index
            * temp_penalty
            * hum_penalty
            * density_penalty
        )

        base_weight = 0.3 + 0.045 * age_days
        final_weight_kg = max(0.8, base_weight + 0.9 * growth_factor + random.gauss(0, 0.12))

        feed_intake_kg = max(1.5, (age_days / 42) * (3.0 + 1.2 * growth_factor + random.gauss(0, 0.2)))

        avg_daily_gain_g = (final_weight_kg * 1000) / age_days

        mortality_rate_pct = 2.0
        mortality_rate_pct += max(0, (stocking_density - 10) * 0.6)
        mortality_rate_pct += max(0, abs(temp_c - 24) * 0.15)
        mortality_rate_pct += max(0, abs(humidity_pct - 65) * 0.05)
        mortality_rate_pct -= (vaccine_score * 1.5)
        mortality_rate_pct -= (housing_quality * 1.0)
        mortality_rate_pct -= (management_index * 1.0)
        mortality_rate_pct += random.gauss(0, 0.4)
        mortality_rate_pct = min(max(mortality_rate_pct, 0.5), 12.0)

        fcr = max(1.2, min(3.0, feed_intake_kg / final_weight_kg + random.gauss(0, 0.05)))

        cycles_per_year = 6
        survivors = flock_size * (1 - mortality_rate_pct / 100)
        revenue_batch = survivors * final_weight_kg * sale_price_usd_kg
        feed_cost = flock_size * feed_intake_kg * feed_price_usd_kg
        annual_revenue_usd = (revenue_batch - feed_cost - energy_cost_usd) * cycles_per_year
        annual_revenue_usd = max(0, annual_revenue_usd)

        writer.writerow([
            round(age_days, 2),
            round(chick_weight_g, 2),
            round(temp_c, 2),
            round(humidity_pct, 2),
            round(stocking_density, 2),
            round(vaccine_score, 3),
            round(breed_index, 3),
            round(housing_quality, 3),
            round(feed_protein_pct, 2),
            round(water_quality, 3),
            round(management_index, 3),
            int(flock_size),
            round(feed_price_usd_kg, 3),
            round(sale_price_usd_kg, 3),
            round(energy_cost_usd, 2),
            round(final_weight_kg, 3),
            round(mortality_rate_pct, 2),
            round(avg_daily_gain_g, 2),
            round(feed_intake_kg, 3),
            round(fcr, 3),
            round(annual_revenue_usd, 2)
        ])

print(f'Wrote {n} rows to {out_path}')