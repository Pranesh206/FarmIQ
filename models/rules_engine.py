def get_recommendations(crop, soil_type, weather, sensors, soils):
    recs = {}
    
    # Sowing (same)
    if crop == 'rice' and weather['rainfall'] > 100:
        recs['sowing'] = 'Optimal time: Now (monsoon).'
    else:
        recs['sowing'] = 'Delay sowing; low rainfall.'
    
    # Irrigation (same)
    if sensors['moisture'] < 30:
        recs['irrigation'] = 'Irrigate immediately; soil dry.'
    else:
        recs['irrigation'] = 'No irrigation needed.'
    
    # Fertilizer (same)
    npk = soils.get('npk', [10,10,10])
    recs['fertilizer'] = f'Apply N:{npk[0]}, P:{npk[1]}, K:{npk[2]} based on soil test.'
    
    # Pest (same)
    recs['pest'] = 'Monitor for diseases; use IPM.'
    
    # Climate-resilient (same)
    if weather['temp'] > 35:
        recs['climate'] = 'Use drought-resistant varieties.'
    
    # [NEW DEV] Sustainability: Basic carbon footprint (dummy formula: fertilizer use * temp factor)
    carbon = (npk[0] * 0.5) + (weather['temp'] * 0.2)  # Simple calc
    recs['sustainability'] = {
        'carbon_footprint': round(carbon, 2),
        'tip': f'Reduce by switching to organic: saves {carbon * 0.1:.1f} kg CO2.'
    }
    
    return recs