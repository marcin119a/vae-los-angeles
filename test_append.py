import json
import pandas as pd
acc_results = [
    {'Method': 'Mean', 'Target': 'DNA', 'MSE': 0.0158, 'Pearson r': 0.8878},
    {'Method': 'KNN', 'Target': 'DNA', 'MSE': 0.0090, 'Pearson r': 0.9382},
    {'Method': 'Conditioned KNN', 'Target': 'DNA', 'MSE': 0.0091, 'Pearson r': 0.9376}
]

vae_results = {'MSE': 0.0070, 'Pearson r': 0.9518}
acc_results.append({
    'Method': 'VAE',
    'Target': 'DNA',
    'MSE': vae_results['MSE'],
    'Pearson r': vae_results['Pearson r']
})

print("acc_results length:", len(acc_results))
print([r['Method'] for r in acc_results])
