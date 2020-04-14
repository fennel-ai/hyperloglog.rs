extern crate hyperloglogplus;

extern crate rand;

use std::collections::hash_map::RandomState;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::prelude::*;

use hyperloglogplus::{HyperLogLog, HyperLogLogPF, HyperLogLogPlus};

fn generate_strings(count: usize) -> Vec<String> {
    let mut rng = rand::thread_rng();

    let mut workload: Vec<String> = (0..count)
        .map(|_| format!("- {} - {} -", rng.gen::<u64>(), rng.gen::<u64>()))
        .collect();

    workload.shuffle(&mut rng);

    workload
}

fn bench_add(c: &mut Criterion) {
    let workload = generate_strings(2000);

    macro_rules! bench_impls {
        ($testname:expr, $impl:ident, $precision:expr) => {
            c.bench_function($testname, |b| {
                b.iter(|| {
                    let mut hll: $impl<String, RandomState> =
                        $impl::new($precision, RandomState::new()).unwrap();

                    for val in &workload {
                        hll.add(&val);
                    }
                })
            });
        };
    }

    bench_impls!["hyperloglog_add_p4", HyperLogLogPF, 4];
    bench_impls!["hyperloglog_add_p8", HyperLogLogPF, 8];
    bench_impls!["hyperloglog_add_p16", HyperLogLogPF, 16];

    bench_impls!["hyperloglogplus_add_p4", HyperLogLogPlus, 4];
    bench_impls!["hyperloglogplus_add_p8", HyperLogLogPlus, 8];
    bench_impls!["hyperloglogplus_add_p16", HyperLogLogPlus, 16];
}

fn bench_hyperloglog_count(c: &mut Criterion) {
    let workload = generate_strings(49200);

    macro_rules! bench_impls {
        ($testname:expr, $impl:ident, $precision:expr) => {
            let mut hll: $impl<String, RandomState> =
                $impl::new($precision, RandomState::new()).unwrap();

            for val in &workload {
                hll.add(&val);
            }

            c.bench_function($testname, |b| {
                b.iter(|| {
                    let val = hll.count();
                    black_box(val);
                })
            });
        };
    }

    bench_impls!["hyperloglog_count_p4", HyperLogLogPF, 4];
    bench_impls!["hyperloglog_count_p8", HyperLogLogPF, 8];
    bench_impls!["hyperloglog_count_p16", HyperLogLogPF, 16];

    bench_impls!["hyperloglogplus_count_p4", HyperLogLogPlus, 4];
    bench_impls!["hyperloglogplus_count_p8", HyperLogLogPlus, 8];
    bench_impls!["hyperloglogplus_count_p16", HyperLogLogPlus, 16];
}

criterion_group!(benches, bench_add, bench_hyperloglog_count);

criterion_main!(benches);