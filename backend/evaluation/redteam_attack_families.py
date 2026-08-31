"""
Red-Team Attack Families for PayShieldAI
Defines the 8 locked adversarial attack patterns using ONLY the canonical 7-feature schema.
"""

import numpy as np
import pandas as pd

def attack_velocity_spike(X_base, y_base):
    """Simulates rapid successive transactions by inflating velocity metrics."""
    X_adv = X_base.copy()
    X_adv['velocity_1h'] = X_adv['velocity_1h'] * 5
    X_adv['velocity_24h'] = X_adv['velocity_24h'] * 3
    return X_adv, y_base

def attack_geographic_spoof(X_base, y_base):
    """Simulates location spoofing or stolen cards used in distant locations."""
    X_adv = X_base.copy()
    X_adv['distance_km'] = X_adv['distance_km'].apply(lambda x: max(x, 5000.0))
    return X_adv, y_base

def attack_coordinated_swarm(X_base, y_base):
    """Slightly elevates multiple risk indicators to stay under individual threshold radars."""
    X_adv = X_base.copy()
    X_adv['velocity_1h'] = X_adv['velocity_1h'] * 1.5
    X_adv['distance_km'] = X_adv['distance_km'] * 1.5
    X_adv['merchant_risk'] = X_adv['merchant_risk'] * 1.2
    return X_adv, y_base

def attack_high_value_cashing(X_base, y_base):
    """Fraudsters attempting to cash out max value in a single hit while keeping velocity zero."""
    X_adv = X_base.copy()
    X_adv['amount'] = X_adv['amount'] * 10
    X_adv['velocity_1h'] = 0.0 
    return X_adv, y_base

def attack_off_hour_strike(X_base, y_base):
    """Exploiting time-based models by forcing transactions into typical sleep hours."""
    X_adv = X_base.copy()
    X_adv['hour'] = 3.0  # 3 AM
    return X_adv, y_base

def attack_new_device_fraud(X_base, y_base):
    """Simulates account takeovers using completely new/unrecognized hardware."""
    X_adv = X_base.copy()
    X_adv['device_age_days'] = 0.0
    return X_adv, y_base

def attack_merchant_compromise(X_base, y_base):
    """Funneling transactions through highly risky or newly set up fraud merchant accounts."""
    X_adv = X_base.copy()
    X_adv['merchant_risk'] = 0.99 
    return X_adv, y_base

def attack_micro_structuring(X_base, y_base):
    """Testing minimum-amount evasion (card testing) with high frequency."""
    X_adv = X_base.copy()
    X_adv['amount'] = 1.05  # Micro-transaction
    X_adv['velocity_1h'] = X_adv['velocity_1h'] * 10
    return X_adv, y_base

def get_redteam_attacks():
    """
    Returns the exact 8 locked attack families required for Stage 9 validation.
    """
    return {
        "Velocity Spike": attack_velocity_spike,
        "Geographic Spoof": attack_geographic_spoof,
        "Coordinated Swarm": attack_coordinated_swarm,
        "High-Value Cashing": attack_high_value_cashing,
        "Off-Hour Strike": attack_off_hour_strike,
        "New Device Fraud": attack_new_device_fraud,
        "Merchant Compromise": attack_merchant_compromise,
        "Micro-Structuring": attack_micro_structuring
    }