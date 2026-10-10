# POC JEPA pour la robotique (Isaac Lab 3.0, DGX Spark)

## Résultats (10/10/2026, DGX Spark, Isaac Lab 3.0 beta2)

Tâche : Cartpole piloté uniquement par caméra, PPO (rl_games), 64 envs, 100 epochs, 1 seed.
Encodeur gelé -> vecteur de features -> politique MLP.

| Encodeur | Reward @ ~100k images | Temps |
|---|---|---|
| ResNet18 (ImageNet) | +1.13 | 268 s |
| V-JEPA 2 ViT-L (vidéo) | -0.33 | 670 s |
| CNN appris de zéro | ~ -2.0 | 326 s |

Référence : cartpole avec état direct (sans caméra) ≈ +4.9.

Observations :
- Un encodeur pré-entraîné aide beaucoup vs CNN de zéro.
- V-JEPA < ResNet18 ici, et ~2.5x plus lent. Hypothèse : le mean-pooling des tokens
  détruit l'info de position (angle du poteau, position du chariot).

Prochaines étapes : pooling spatial (grille 4x4), 2 frames consécutives, 3 seeds.
