import pandas as pd
acc_results = [
    {'Method': 'Mean', 'Target': 'DNA', 'MSE': 0.0158, 'Pearson r': 0.8878},
    {'Method': 'KNN', 'Target': 'DNA', 'MSE': 0.0090, 'Pearson r': 0.9382},
    {'Method': 'Conditioned KNN', 'Target': 'DNA', 'MSE': 0.0091, 'Pearson r': 0.9376},
    {'Method': 'VAE', 'Target': 'DNA', 'MSE': 0.0070, 'Pearson r': 0.9518}
]

acc_df = pd.DataFrame(acc_results)
print("DataFrame Methods list:", acc_df['Method'].tolist())

# Check MIMIR logic
mimir_results = {'MSE': 0.0065, 'Pearson r': 0.96}
acc_results.append({
    'Method': 'MIMIR',
    'Target': 'DNA',
    'MSE': mimir_results['MSE'],
    'Pearson r': mimir_results['Pearson r']
})

acc_df = pd.DataFrame(acc_results)
print("DataFrame with MIMIR Methods list:", acc_df['Method'].tolist())
