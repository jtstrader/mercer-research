use criterion::{criterion_group, criterion_main, Criterion};
use mercer_research::{
    utils::kernel::{Padding, Pooling},
    RCNLayer, RCN,
};

pub fn train_model(c: &mut Criterion) {
    c.bench_function("Train w/ 10 Epochs, 10 in Batch, 100 CSL", |b| {
        let mut model = RCN::new(
            10,
            &[
                RCNLayer::Convolve2D(Padding::Same),
                RCNLayer::Pool2D(Pooling::Max),
                RCNLayer::Convolve2D(Padding::Same),
                RCNLayer::Pool2D(Pooling::Max),
            ],
            &[30],
            "images\\mnist_png\\train",
            "images\\mnist_png\\valid",
        );
        b.iter(|| model.train(10, 50, 3_f64, 200).unwrap());
    });
}

criterion_group!(benches, train_model);
criterion_main!(benches);
