use mercer_research::{
    utils::kernel::{Padding, Pooling},
    RCNLayer, RCN,
};

fn main() {
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

    model.train(10, 100, 3_f64, 500).unwrap();
}
