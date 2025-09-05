#!/usr/bin/env python3
import argparse
import os
import glob
import numpy as np
import pandas as pd
import hashlib


def find_csv(folder: str) -> str:
    pattern = os.path.join(folder, "*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"Nessun file CSV trovato in {folder}")
    files.sort()
    return files[0]


def load_df_standard(path):
    return pd.read_csv(
        path,
        header=None,
        skiprows=1,
        names=['timestamp', 'seq_num'],
        dtype={'timestamp': 'float64', 'seq_num': 'int64'}
    )


def load_4_file(path):
    """Load the merged dataset CSV file with proper column handling"""
    df = pd.read_csv(path)
    
    if df.empty:
        raise ValueError(f"Il file {path} è vuoto o non contiene dati validi.")
    
    # Ensure we have a 'time' column for internal processing
    # but don't rename 'timestamp' since downstream scripts expect it
    if 'timestamp' in df.columns and 'time' not in df.columns:
        df['time'] = df['timestamp']  # Create time column for internal use
    
    return df


def save_df(path, df):
    dirname = os.path.dirname(path)
    if dirname:  # Only create directory if path has a directory component
        os.makedirs(dirname, exist_ok=True)
    df.to_csv(path, index=False, header=True)


def build_intervals(df1: pd.DataFrame, df2: pd.DataFrame):
    """
    Ricostruisce gli intervalli (start_time, end_time) per ogni seq_num.
    Mantiene esattamente la stessa logica del codice originale.
    Restituisce interval_df e l'insieme to_remove.
    """
    # Conteggi e rimozioni iniziali (vettoriale)
    vc1 = df1['seq_num'].value_counts()
    vc2 = df2['seq_num'].value_counts()
    aligned_vc2 = vc2.reindex(vc1.index).fillna(0).astype('int64')
    to_remove = set(vc1.index[(vc1 > 1) & (vc1 > aligned_vc2 + 1)])

    # Pre-ordino e raggruppo una volta sola
    df1s = df1.sort_values(['seq_num', 'timestamp'], kind='mergesort')
    df2s = df2.sort_values(['seq_num', 'timestamp'], kind='mergesort')

    starts_grp = df1s.groupby('seq_num', sort=False)['timestamp']
    ends_grp   = df2s.groupby('seq_num', sort=False)['timestamp']

    # Materializzo in array NumPy (accesso O(1) per seq)
    starts_map = {k: v.to_numpy(copy=False) for k, v in starts_grp}
    ends_map   = {k: v.to_numpy(copy=False) for k, v in ends_grp}

    all_seqs = set(starts_map.keys()) | set(ends_map.keys())

    out_seq = []
    out_start = []
    out_end = []

    # Loop per-seq efficiente (niente .loc ripetuti)
    for seq in all_seqs:
        if seq in to_remove:
            continue

        s = starts_map.get(seq, np.array([], dtype='float64'))
        e = ends_map.get(seq, np.array([], dtype='float64'))
        ns, ne = s.size, e.size

        if ns == 0 or ne == 0:
            to_remove.add(seq)
            continue

        #Qua prova poi a ricostruire (laddove possibile) con ttl, ma per ora scarta
        # Casi come nell'originale
        #if ns > 1 and ne == 1:
        #    # multiple start, single end
        #    out_seq.extend([seq] * ns)
        #    out_start.extend(s.tolist())
        #    out_end.extend([e[0]] * ns)

        elif ns == 1 and ne == 1:
            # single start, single end
            out_seq.append(seq)
            out_start.append(s[0])
            out_end.append(e[0])

        elif ns > 1 and ne > 1 and ns == ne:
            # multiple start e multiple end con stesso numero di occorrenze
            # Stessa condizione dell'originale: e[i] < s[i+1] per tutti i prefissi
            if np.all(e[:-1] < s[1:]):
                out_seq.extend([seq] * ns)
                out_start.extend(s.tolist())
                out_end.extend(e.tolist())
            else:
                to_remove.add(seq)
        else:
            # qualsiasi altro caso è ambiguo
            to_remove.add(seq)

    interval_df = pd.DataFrame(
        {
            'seq_num': np.array(out_seq, dtype='int64'),
            'start_time': np.array(out_start, dtype='float64'),
            'end_time': np.array(out_end, dtype='float64'),
        },
        columns=['seq_num', 'start_time', 'end_time']
    )
    return interval_df, to_remove


def main():
    parser = argparse.ArgumentParser(
        description='Add out-of-order features to dataset using extracted OOO_SEG data.'
    )
    # Match the pipeline call format: --input-file and --output-file
    parser.add_argument('--input-file', required=True,
                        help='Input CSV file (collision_filtered.csv)')
    parser.add_argument('--output-file', required=True,
                        help='Output CSV file (ooo_enhanced.csv)')
    
    # Optional directories for OOO data (use defaults if not specified)
    parser.add_argument('--dir1', default='results_1G/SND_TS_SEQ_NUM',
                        help='Directory with send timestamp CSV files')
    parser.add_argument('--dir2', default='results_1G/RCV_TS_SEQ_NUM',
                        help='Directory with receive timestamp CSV files')
    parser.add_argument('--dir3', default='results_1G/OOO_SEG',
                        help='Directory with out-of-order segment CSV files')
    args = parser.parse_args()

    # Check if required directories exist
    if not os.path.exists(args.dir1):
        raise FileNotFoundError(f"SND_TS_SEQ_NUM directory not found: {args.dir1}")
    if not os.path.exists(args.dir2):
        raise FileNotFoundError(f"RCV_TS_SEQ_NUM directory not found: {args.dir2}")
    if not os.path.exists(args.dir3):
        raise FileNotFoundError(f"OOO_SEG directory not found: {args.dir3}")

    # Find the CSV files
    path1 = find_csv(args.dir1)
    path2 = find_csv(args.dir2)
    path3 = find_csv(args.dir3)
    input_file = args.input_file

    print(f"Using SND_TS_SEQ_NUM: {path1}")
    print(f"Using RCV_TS_SEQ_NUM: {path2}")
    print(f"Using OOO_SEG: {path3}")
    print(f"Using input dataset: {input_file}")

    # Load the timestamp files
    df1 = load_df_standard(path1)
    df2 = load_df_standard(path2)

    # Build intervals using send and receive timestamps
    interval_df, to_remove = build_intervals(df1, df2)
    print(f"Built intervals for {len(interval_df)} packets, removing {len(to_remove)} problematic seq_nums")

    # Load OOO segments file
    df3 = load_df_standard(path3)
    df3_f = df3[~df3['seq_num'].isin(to_remove)]
    print(f"Loaded {len(df3)} OOO segments, {len(df3_f)} after filtering")

    # Verify: ogni riga di df3_f deve matchare una e UNA sola riga di interval_df
    merged3 = df3_f.merge(
        interval_df,
        left_on=['seq_num', 'timestamp'],
        right_on=['seq_num', 'end_time'],
        how='left',
        indicator=True
    )
    if merged3['_merge'].eq('both').sum() != len(df3_f):
        print(f"Warning: Only {merged3['_merge'].eq('both').sum()} of {len(df3_f)} OOO segments matched intervals")

    # 'ooo' vettoriale: 1 se (seq_num, end_time) compare in df3_f
    matches = merged3.loc[merged3['_merge'] == 'both', ['seq_num', 'end_time']].drop_duplicates()
    interval_df = interval_df.merge(
        matches.assign(ooo=1),
        on=['seq_num', 'end_time'],
        how='left'
    )
    interval_df['ooo'] = interval_df['ooo'].fillna(0).astype('int8')
    print(f"Marked {interval_df['ooo'].sum()} packets as out-of-order")

    # Load the input dataset (collision-filtered data)
    df4 = load_4_file(input_file)
    df4_f = df4[~df4['seq_num'].isin(to_remove)]
    print(f"Loaded input dataset: {len(df4)} packets, {len(df4_f)} after filtering removed seq_nums")

    # Join finale: join on seq_num and filter for time within the interval
    merged = pd.merge(df4_f, interval_df, on='seq_num', how='left')
    
    # Filter to keep only packets whose timestamp falls within their start/end interval
    valid_time_mask = (merged['time'] > merged['start_time']) & (merged['time'] < merged['end_time'])
    final_ds = merged[valid_time_mask]
    
    # Drop the internal 'time' column if it was created (keep original 'timestamp')
    if 'time' in final_ds.columns and 'timestamp' in final_ds.columns:
        final_ds = final_ds.drop(columns=['time'])
    
    print(f"Final dataset: {len(final_ds)} packets (from {len(merged)} after interval filtering)")
    print(f"OOO packets in final dataset: {final_ds['ooo'].sum()}")

    # Save the enhanced dataset
    save_df(args.output_file, final_ds)
    print(f"Saved OOO-enhanced dataset to: {args.output_file}")


if __name__ == '__main__':
    main()