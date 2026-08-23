# PayShield AI — Development Rules

## Workflow
1. Every feature gets its own module.
2. Every feature must have tests.
3. Never use real customer/payment data.
4. Use synthetic data only.
5. Never fabricate metrics.
6. Keep the core pipeline working even if advanced features fail.
7. Commit working milestones frequently.

## Core Pipeline
Identify → Generate → Defend → Evaluate → Evolve → Retrain

## Code Quality
- Python 3.13
- Type hints where practical
- Clear module boundaries
- Reproducible experiments
- Configuration through files/environment variables
