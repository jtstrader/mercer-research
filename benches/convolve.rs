// Benchmark convolution functions to see what is more efficient

use criterion::{criterion_group, criterion_main, Criterion};
use image::io::Reader as ImageReader;
use mercer_research::get_pixel_matrix;
use mercer_research::utils::kernel::{Convolve2D, SeparableOperator, __TOP_SOBEL};
use nalgebra::DMatrix;

fn get_image_data() -> DMatrix<i16> {
    get_pixel_matrix(
        &ImageReader::open("images\\mnist_png\\train\\4\\2.png")
            .unwrap()
            .decode()
            .unwrap(),
    )
    .unwrap()
}

pub fn simple_convolution(c: &mut Criterion) {
    c.bench_function("Simple Convolution", |b| {
        let img_data: DMatrix<i16> = get_image_data();
        b.iter(|| img_data.convolve_2d(&__TOP_SOBEL));
    });
}

pub fn separated_convolution(c: &mut Criterion) {
    c.bench_function("Separated Convolution", |b| {
        let img_data: DMatrix<i16> = get_image_data();
        b.iter(|| img_data.convolve_2d_separated(SeparableOperator::Top));
    });
}

criterion_group!(benches, simple_convolution, separated_convolution);
criterion_main!(benches);
