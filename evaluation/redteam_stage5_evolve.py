"""
STAGE 5: EVOLVE (Adversarial Attack Generation)
- Uses the trained model as a fitness function.
- Evolves attacks to minimize the model's fraud probability.
- Measures the model's performance on these evolved (harder) attacks.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings("ignore")

from generation.generator import FEATURES
from evaluation.hard_benchmark_stage2_repaired import (
    generate_stage2_repaired_benchmark,
)
from evaluation.redteam_stage2c_retest import make_unseen_attacks


def load_or_train_model():
    """Try to load saved model, otherwise train a fresh one."""
    try:
        model = joblib.load("models/xgboost_detector_retrained.joblib")
        print("Loaded model from models/xgboost_detector_retrained.joblib")
        return model
    except FileNotFoundError:
        print("Model not found, training a fresh one...")
        train_df = generate_stage2_repaired_benchmark(n=10000, fraud_rate=0.05, seed=42)
        X_train, _, y_train, _ = train_test_split(
            train_df[FEATURES],
            train_df["is_fraud"],
            test_size=0.30,
            random_state=42,
            stratify=train_df["is_fraud"],
        )
        model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(X_train, y_train)
        joblib.dump(model, "models/xgboost_detector_retrained.joblib")
        return model


def get_benign_baseline(n=500, seed=42):
    """Generate benign transactions to use as starting points for evolution."""
    df = generate_stage2_repaired_benchmark(n=n, fraud_rate=0.0, seed=seed)
    return df


def mutate_attack(row, rng, mutation_rate=0.3):
    """Apply random mutations to a single transaction (attack)."""
    row = row.copy()
    
    # Mutate amount (scale up or down slightly)
    if rng.random() < mutation_rate:
        row["amount"] *= rng.uniform(0.8, 1.5)
        row["amount"] = max(1, row["amount"])  # keep positive
    
    # Mutate velocity_1h
    if rng.random() < mutation_rate:
        row["velocity_1h"] += rng.integers(-2, 3)
        row["velocity_1h"] = max(0, row["velocity_1h"])
    
    # Mutate velocity_24h
    if rng.random() < mutation_rate:
        row["velocity_24h"] += rng.integers(-3, 4)
        row["velocity_24h"] = max(0, row["velocity_24h"])
    
    # Mutate distance_km
    if rng.random() < mutation_rate:
        row["distance_km"] += rng.uniform(-20, 40)
        row["distance_km"] = max(0, row["distance_km"])
    
    # Mutate merchant_risk (slightly)
    if rng.random() < mutation_rate:
        row["merchant_risk"] += rng.uniform(-0.1, 0.1)
        row["merchant_risk"] = np.clip(row["merchant_risk"], 0, 1)
    
    # Mutate hour (shift by a few hours)
    if rng.random() < mutation_rate:
        row["hour"] = (row["hour"] + rng.integers(-3, 4)) % 24
    
    # Mutate device_age_days (slightly)
    if rng.random() < mutation_rate:
        row["device_age_days"] += rng.integers(-50, 100)
        row["device_age_days"] = max(1, row["device_age_days"])
    
    return row


def evolve_attacks(model, base_benign, n_attacks=500, generations=10, population_size=100, seed=999):
    """
    Genetic algorithm to evolve stealthy attacks.
    Fitness = lower predicted fraud probability (we want to minimize it).
    """
    rng = np.random.default_rng(seed)
    
    # Step 1: Create initial population by injecting attack patterns into benign rows
    # We take the benign rows and apply the standard unseen attack transformation first.
    # Then we mutate them further.
    
    # Pick random benign rows
    idx = rng.choice(base_benign.index, size=n_attacks, replace=False)
    population = base_benign.loc[idx].copy()
    
    # Apply initial attack injection (make them fraudulent in pattern)
    # We'll use the same logic as make_unseen_attacks but only on these selected rows
    fraud_indices = rng.choice(population.index, size=min(n_attacks, len(population)), replace=False)
    # Assign fraud labels for evolution (we will keep them as fraud, just try to make them stealthier)
    
    # But we need to start with visible attack patterns to evolve from.
    # Let's apply the three attack families to the selected rows.
    # Split into thirds for each family
    split1 = n_attacks // 3
    split2 = 2 * n_attacks // 3
    
    indices = list(population.index)
    rng.shuffle(indices)
    
    # Velocity family
    idx_vel = indices[:split1]
    population.loc[idx_vel, "velocity_1h"] += rng.integers(2, 5, len(idx_vel))
    population.loc[idx_vel, "velocity_24h"] += rng.integers(3, 10, len(idx_vel))
    
    # Geographic family
    idx_geo = indices[split1:split2]
    population.loc[idx_geo, "distance_km"] += rng.uniform(30, 120, len(idx_geo))
    
    # Coordinated family
    idx_coord = indices[split2:]
    population.loc[idx_coord, "amount"] *= rng.uniform(1.5, 3.0, len(idx_coord))
    population.loc[idx_coord, "velocity_24h"] += rng.integers(2, 8, len(idx_coord))
    population.loc[idx_coord, "distance_km"] += rng.uniform(20, 100, len(idx_coord))
    
    # Label all as fraud (we are evolving attacks)
    population["is_fraud"] = 1
    
    print(f"Initial population size: {len(population)}")
    
    # We'll evolve the entire population as a pool.
    # For simplicity, we use a (mu, lambda) style: keep the top performers.
    
    best_population = population.copy()
    
    for gen in range(generations):
        # Create offspring by mutating the current best population
        offspring_list = []
        for _ in range(population_size):
            # Pick a random parent from the best population
            parent_idx = rng.choice(best_population.index)
            parent = best_population.loc[parent_idx]
            child = mutate_attack(parent, rng, mutation_rate=0.3)
            child["is_fraud"] = 1
            offspring_list.append(child)
        
        offspring_df = pd.DataFrame(offspring_list)
        
        # Combine current best with offspring (elitism)
        combined = pd.concat([best_population, offspring_df], ignore_index=True)
        
        # Predict fraud probabilities for all candidates
        probs = model.predict_proba(combined[FEATURES])[:, 1]
        combined["fraud_prob"] = probs
        
        # Select the n_attacks individuals with the LOWEST fraud probability (best evaders)
        combined_sorted = combined.sort_values("fraud_prob", ascending=True)
        best_population = combined_sorted.head(n_attacks).drop(columns=["fraud_prob"]).copy()
        
        # Optional: print progress
        avg_prob = best_population["fraud_prob"].mean() if "fraud_prob" in best_population else 0
        if gen % 2 == 0:
            print(f"Generation {gen+1}/{generations} - Avg fraud prob of best: {avg_prob:.4f}")
    
    # Ensure all are labelled fraud (they are attacks)
    best_population["is_fraud"] = 1
    
    return best_population


def evaluate_evolved(model, test_df, evolved_df):
    """Compare performance on standard test vs evolved attacks."""
    
    # Standard test set (mixed attacks)
    X_std = test_df[FEATURES]
    y_std = test_df["is_fraud"]
    probs_std = model.predict_proba(X_std)[:, 1]
    auc_std = roc_auc_score(y_std, probs_std)
    rec_std = recall_score(y_std, (probs_std >= 0.5).astype(int), zero_division=0)
    
    # Evolved attacks (should be stealthier)
    X_evol = evolved_df[FEATURES]
    y_evol = evolved_df["is_fraud"]  # All are 1
    probs_evol = model.predict_proba(X_evol)[:, 1]
    # Compute metrics on the evolved set
    # Since y_evol is all 1s, AUC is not meaningful (needs negative class).
    # Instead, we compare the average predicted probability of fraud.
    avg_prob_evol = probs_evol.mean()
    # Also compute recall at threshold 0.5 (if threshold applied)
    preds_evol = (probs_evol >= 0.5).astype(int)
    recall_evol = recall_score(y_evol, preds_evol, zero_division=0)
    
    return {
        "standard_auc": auc_std,
        "standard_recall": rec_std,
        "evolved_avg_prob": avg_prob_evol,
        "evolved_recall_at_05": recall_evol,
        "evolved_probs": probs_evol,
    }


if __name__ == "__main__":
    print("=== STAGE 5: EVOLVE (Adversarial Attack Generation) ===")
    
    # 1. Load model
    model = load_or_train_model()
    
    # 2. Generate a standard test set (for comparison)
    print("\nGenerating standard test set...")
    test_df = generate_stage2_repaired_benchmark(n=5000, fraud_rate=0.05, seed=999)
    test_df = make_unseen_attacks(test_df, seed=12345)
    print(f"Standard test size: {len(test_df)} rows")
    print(f"Fraud cases: {int(test_df['is_fraud'].sum())}")
    
    # 3. Get benign starting points for evolution
    benign = get_benign_baseline(n=1000, seed=42)
    
    # 4. Evolve stealthy attacks
    print("\n--- Starting Evolution (this may take a moment) ---")
    evolved = evolve_attacks(
        model,
        benign,
        n_attacks=500,
        generations=15,
        population_size=200,
        seed=999
    )
    print(f"\nEvolved attacks generated: {len(evolved)}")
    
    # 5. Evaluate the model on evolved attacks
    results = evaluate_evolved(model, test_df, evolved)
    
    print("\n" + "=" * 60)
    print("EVOLUTION RESULTS")
    print("=" * 60)
    print(f"Standard Test AUC (mixed attacks): {results['standard_auc']:.4f}")
    print(f"Standard Test Recall @ 0.5:       {results['standard_recall']:.4f}")
    print()
    print(f"Evolved Attacks Average Fraud Prob: {results['evolved_avg_prob']:.4f}")
    print(f"Evolved Attacks Recall @ 0.5:       {results['evolved_recall_at_05']:.4f}")
    print("=" * 60)
    
    # 6. Save evolved attacks for Stage 6 (Retrain)
    evolved.to_csv("data/evolved_attacks.csv", index=False)
    print("\nEvolved attacks saved to data/evolved_attacks.csv")
    
    # 7. Verdict
    print("\nDIAGNOSIS:")
    if results['evolved_avg_prob'] < 0.20:
        print("STATUS: EVOLUTION SUCCESSFUL — Model is easily fooled.")
        print("        Average fraud probability on evolved attacks is low.")
        print("        RETRAINING REQUIRED (Stage 6).")
    else:
        print("STATUS: EVOLUTION WEAK — Model remains robust to evolved attacks.")
        print("        Model may not need immediate retraining.")