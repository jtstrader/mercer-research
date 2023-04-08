use criterion::{criterion_group, criterion_main, Criterion};
use rcn::{
    rcn::RCNLayer,
    rcn::RCN,
    utils::kernel::{Padding, Pooling},
};

pub fn train_model(c: &mut Criterion) {
    c.bench_function("Train w/ 10 Epochs, 10 in Batch, 100 CSL", |b| {
        let mut model = RCN::new(
            10,
            vec![
                RCNLayer::Convolve2D(Padding::Same),
                RCNLayer::Pool2D(Pooling::Max),
                RCNLayer::Convolve2D(Padding::Same),
                RCNLayer::Pool2D(Pooling::Max),
            ],
            vec![30],
            "images/mnist_png/training",
            "images/mnist_png/testing",
        );
        b.iter(|| model.train(10, 50, 3_f64, 500, 500).unwrap());
    });
}

criterion_group!(benches, train_model);
criterion_main!(benches);
