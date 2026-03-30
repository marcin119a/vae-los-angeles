import pandas as pd
import kagglehub

dna_path = kagglehub.dataset_download('martininf1n1ty/dna-methylation-final-adnotated')
df = pd.read_parquet(f'{dna_path}/part-00000-db52fd1e-039e-43fd-9eef-5f241ff75754-c000.snappy.parquet')
df_sorted = df.sort_values(by='probe_id_id')
grouped = df_sorted.groupby('case_barcode')['probe_id_id'].apply(list).reset_index()
valid_cases = grouped[grouped['probe_id_id'].apply(len) == 572]

if not valid_cases.empty:
    probes_list = valid_cases['probe_id_id'].iloc[0]
    with open('data/cpg_probes_572.txt', 'w') as f:
        for p in probes_list:
            f.write(f'{p}\n')
    print('Saved 572 probes to data/cpg_probes_572.txt')
else:
    print('No cases found with exactly 572 probes.')
