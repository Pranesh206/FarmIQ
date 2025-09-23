#!/usr/bin/env python3

import json

import pandas as pd

import numpy as np

import argparse

import os

from datetime import datetime, timedelta



def generate_farmers(num_farmers=10, output_dir='data'):

    crops = ['rice', 'wheat', 'cotton', 'maize']

    locations = ['Punjab', 'UP', 'Maharashtra', 'Gujarat', 'Kerala']

    soil_types = ['loamy', 'sandy', 'clay', 'alluvial']

    

    farmers = []

    for I in range(1, num_farmers + 1):

        farmer = {

            'id': I,

            'username': f'farmer{I}',

            'crop': np.random.choice(crops),

            'location': np.random.choice(locations),

            'soil_type': np.random.choice(soil_types),

            'farm_size': round(np.random.uniform(2, 10), 1),

            'history': [f"2023: Yield {round(np.random.uniform(2, 6), 1)} tons/ha"],

            'consent_agristack': np.random.choice([True, False], p=[0.8, 0.2])

        }

        farmers.append(farmer)

    

    with open(os.path.join(output_dir, 'dummy_farmers.json'), 'w') as f:

        json.dump(farmers, f, indent=2)

    print(f"[DATA] Generated {num_farmers} farmers.")



def generate_weather(num_locations=5, output_dir='data'):

    locations = ['Punjab', 'UP', 'Maharashtra', 'Gujarat', 'Kerala']

    weather_data = {}

    for loc in locations[:num_locations]:
        # Example: Generate random weather data for each location
        weather_data[loc] = {
            'temperature': round(np.random.uniform(20, 40), 1),
            'humidity': round(np.random.uniform(40, 90), 1),
            'rainfall': round(np.random.uniform(0, 200), 1)
        }