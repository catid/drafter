"""Exact, bounded-memory statistics for all offline sparse lookup events.

Each committed optimizer batch owns one reduced Parquet chunk. DuckDB merges
keys with disk spilling; no Python dictionary grows with vocabulary pairs.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from .sparse import hash_keys


def pack_keys(d, previous, current, vocab_size):
    return (d * vocab_size + previous) * vocab_size + current


def unpack_keys(keys, vocab_size):
    return keys // (vocab_size * vocab_size), (keys // vocab_size) % vocab_size, keys % vocab_size


class SparseAccessRecorder:
    def __init__(self, directory, identity, vocab_size):
        self.directory = Path(directory) / identity
        self.directory.mkdir(parents=True, exist_ok=True)
        self.vocab_size = vocab_size

    def write_batch(self, packed_keys, epoch, batch):
        import pyarrow as pa
        import pyarrow.parquet as pq

        values = packed_keys.detach().cpu().numpy().astype(np.int64, copy=False)
        unique, counts = np.unique(values, return_counts=True)
        path = self.directory / f'epoch{epoch:03d}-batch{batch:08d}.parquet'
        fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
        os.close(fd)
        try:
            pq.write_table(pa.table({'packed_key': unique, 'frequency': counts.astype(np.uint64)}), temporary,
                           compression='zstd')
            with open(temporary, 'rb') as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return {'path': str(path.resolve()), 'events': len(values), 'unique_batch_keys': len(unique),
                'epoch': epoch, 'batch': batch,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    def finalize(self, chunks, sparse, *, memory_limit='512MB', benchmark_entries=1_000_000,
                 benchmark_queries=4096, seed=2026):
        import duckdb

        paths = [item['path'] for item in chunks]
        if len(paths) != len(set(paths)):
            raise ValueError('Duplicate sparse batch chunks would double-count lookup events')
        for item in chunks:
            if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('Sparse statistics chunk changed after its optimizer checkpoint')
        if not paths:
            return {'status': 'not_run', 'reason': 'no sparse-stage lookup events'}
        database = self.directory / 'exact_key_counts.duckdb'
        connection = duckdb.connect(str(database))
        connection.execute('SET memory_limit = ?', [memory_limit])
        connection.execute('SET temp_directory = ?', [str(self.directory / 'spill')])
        connection.execute('SET threads = 2')
        connection.execute('SET preserve_insertion_order = false')
        connection.execute('CREATE OR REPLACE TABLE exact_key_counts AS SELECT packed_key, '
                           'SUM(frequency)::UBIGINT AS frequency FROM read_parquet(?) GROUP BY packed_key', [paths])
        V = self.vocab_size
        # This SQL is exactly sparse.hash_keys' deterministic bucket hash.
        bucket = (f'((((((packed_key // {V * V}) * 73856093 + '
                  f'((packed_key // {V}) % {V}) * 19349663 + '
                  f'(packed_key % {V}) * 83492791 + 104729) % 2147483647) '
                  f'* 48271 + 1) % 2147483647) % {sparse.capacity})')
        connection.execute('CREATE OR REPLACE TABLE bucket_counts AS SELECT ' + bucket +
                           ' AS bucket, COUNT(*)::UBIGINT AS distinct_keys, SUM(frequency)::UBIGINT AS frequency '
                           'FROM exact_key_counts GROUP BY bucket')
        unique, events = connection.execute('SELECT COUNT(*),SUM(frequency) FROM exact_key_counts').fetchone()
        occupied, collisions, weighted, colliding_keys = connection.execute(
            'SELECT COUNT(*), SUM(distinct_keys-1), SUM(CASE WHEN distinct_keys>1 THEN frequency ELSE 0 END), '
            'SUM(CASE WHEN distinct_keys>1 THEN distinct_keys ELSE 0 END) FROM bucket_counts').fetchone()
        if events != sum(c['events'] for c in chunks):
            raise RuntimeError('External sparse-key merge lost or duplicated events')
        # Deterministic hash sampling spans offsets/tokens, rather than taking a
        # low-ID lexical prefix of the enormous observed pair table.
        selected = connection.execute(
            'SELECT packed_key,frequency FROM exact_key_counts ORDER BY hash(packed_key) LIMIT ?',
            [min(benchmark_entries, unique)]).fetchnumpy()
        exact_keys = torch.from_numpy(selected['packed_key'].astype(np.int64)).to(sparse.values.device)
        frequencies = torch.from_numpy(selected['frequency'].astype(np.float64))
        permutation = exact_keys.argsort()
        exact_keys = exact_keys[permutation]
        frequencies = frequencies[permutation.cpu()]
        d, previous, current = unpack_keys(exact_keys, V)
        exact_values = sparse.lookup(d, previous, current).detach()
        generator = torch.Generator().manual_seed(seed)
        sampled_indices = torch.multinomial(frequencies, benchmark_queries, replacement=True, generator=generator)
        queries = exact_keys[sampled_indices.to(exact_keys.device)]
        benchmark = lookup_benchmark(sparse, exact_keys, exact_values, queries, V)
        connection.close()
        return {'status': 'measured', 'coverage': 'every lookup in every committed offline sparse-stage optimizer batch',
                'capacity': sparse.capacity, 'unique_observed_keys': unique, 'observed_lookup_events': events,
                'load_factor_unique_keys_per_bucket': unique / sparse.capacity,
                'occupied_buckets': occupied, 'occupancy_fraction': occupied / sparse.capacity,
                'collision_count': collisions, 'distinct_keys_in_collided_buckets': colliding_keys,
                'frequency_weighted_collision_count': weighted,
                'collision_definition': 'sum over buckets of max(distinct observed keys minus one, zero)',
                'frequency_weighted_definition': 'sum of all lookup frequencies in buckets containing more than one distinct key',
                'raw_parameter_bytes': sparse.values.numel() * sparse.values.element_size(),
                'nonzero_parameter_value_bytes': int(sparse.values.count_nonzero()) * sparse.values.element_size(),
                'exact_key_encoding': '(d*V+previous)*V+current; offsets d start at one',
                'global_aggregation': 'DuckDB exact grouped sum with disk spill', 'memory_limit': memory_limit,
                'committed_chunks': len(chunks), 'database': str(database),
                'database_bytes': database.stat().st_size, 'chunk_disk_bytes': sum(Path(p).stat().st_size for p in paths),
                'lookup_benchmark': benchmark}


@torch.no_grad()
def lookup_benchmark(sparse, keys, values, queries, vocab_size, repeats=15, warmups=3):
    d, previous, current = unpack_keys(queries, vocab_size)
    def hashed():
        return sparse.lookup(d, previous, current)
    def exact():
        return values[torch.searchsorted(keys, queries)]
    torch.testing.assert_close(hashed(), exact(), rtol=0, atol=0)
    device = queries.device
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    timings = {}
    for name, fn in [('signed_feature_hash', hashed), ('sorted_exact_observed_pairs', exact)]:
        for _ in range(warmups):
            fn()
        sync()
        samples, cuda_samples = [], []
        for _ in range(repeats):
            sync()
            start = time.perf_counter()
            if device.type == 'cuda':
                with torch.cuda.device(device):
                    event_start = torch.cuda.Event(enable_timing=True)
                    event_end = torch.cuda.Event(enable_timing=True)
                    event_start.record()
            fn()
            if device.type == 'cuda':
                event_end.record(torch.cuda.current_stream(device))
            sync()
            samples.append((time.perf_counter() - start) * 1000)
            if device.type == 'cuda':
                cuda_samples.append(event_start.elapsed_time(event_end))
        rng = np.random.default_rng(2026)
        bootstrap = np.median(rng.choice(samples, (1000, len(samples)), replace=True), axis=1)
        timings[name] = {'median_ms': float(np.median(samples)), 'p95_ms': float(np.quantile(samples, .95)),
                         'median_bootstrap_95ci_ms': np.quantile(bootstrap, [.025, .975]).tolist(),
                         'samples_ms': samples,
                         'cuda_event_samples_ms': cuda_samples,
                         'cuda_event_median_ms': float(np.median(cuda_samples)) if cuda_samples else None}
    return {'device': str(device), 'queries_per_repeat': len(queries), 'unique_table_keys': len(keys),
            'exact_table_bytes': keys.numel() * keys.element_size() + values.numel() * values.element_size(),
            'sample_method': 'deterministic hash sample of global distinct keys; frequency-weighted queries within sample',
            'exact_table_values': 'materialized fitted signed-hash values per observed key; lookup-cost diagnostic only, not an independently fitted model',
            'warmups': warmups, 'repeats': repeats, 'synchronized_wall_timing': True,
            'query_values_equal': True, 'timings': timings}
