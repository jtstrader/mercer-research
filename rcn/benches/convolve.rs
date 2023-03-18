// Benchmark convolution functions to see what is more efficient

use criterion::{criterion_group, criterion_main, Criterion};
use image::io::Reader as ImageReader;
use mercer_research::get_pixel_matrix;
use mercer_research::utils::kernel::{
    Convolve2D, Padding, Pool2D, Pooling, SeparableOperator, __TOP_SOBEL,
};
use nalgebra::DMatrix;

fn get_image_data() -> DMatrix<f64> {
    get_pixel_matrix(
        &ImageReader::open("images/mnist_png/training/4/2.png")
            .unwrap()
            .decode()
            .unwrap(),
    )
    .unwrap()
}

pub fn simple_convolution(c: &mut Criterion) {
    c.bench_function("Simple Convolution", |b| {
        let img_data: DMatrix<f64> = get_image_data();
        b.iter(|| img_data.convolve_2d(&__TOP_SOBEL, &Padding::None));
    });
}

pub fn separated_convolution(c: &mut Criterion) {
    c.bench_function("Separated Convolution", |b| {
        let img_data: DMatrix<f64> = get_image_data();
        b.iter(|| img_data.convolve_2d_separated(SeparableOperator::Top, &Padding::None));
    });
}

pub fn padded_simple(c: &mut Criterion) {
    c.bench_function("Simple Convolution (Padded: Same)", |b| {
        let img_data: DMatrix<f64> = get_image_data();
        b.iter(|| img_data.convolve_2d(&__TOP_SOBEL, &Padding::Same));
    });
}

pub fn padded_separated(c: &mut Criterion) {
    c.bench_function("Separated Convolution (Padded: Same)", |b| {
        let img_data: DMatrix<f64> = get_image_data();
        b.iter(|| img_data.convolve_2d_separated(SeparableOperator::Top, &Padding::Same));
    });
}

pub fn max_pooling(c: &mut Criterion) {
    c.bench_function("Max Pooling", |b| {
        let img_data: DMatrix<f64> = get_image_data();
        b.iter(|| img_data.pool_2d(&Padding::Same, &Pooling::Max));
    });
}

criterion_group!(
    benches,
    simple_convolution,
    separated_convolution,
    padded_simple,
    padded_separated,
    max_pooling
);
criterion_main!(benches);
