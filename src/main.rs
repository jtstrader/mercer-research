use clap::Parser;
use mercer_research::{
    utils::kernel::{Padding, Pooling},
    RCNLayer, RCN,
};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number of classes
    #[arg(short, long, default_value_t = 10)]
    num_classes: usize,

    /// Training directory
    #[arg(long, default_value_t = String::from("images/mnist_png/training"))]
    training_path: String,

    /// Testing/validation directory
    #[arg(long, default_value_t = String::from("images/mnist_png/testing"))]
    testing_path: String,

    /// Number of items to train on per class
    #[arg(long, default_value_t = 500)]
    training_class_size: usize,

    /// Number of items to test on per class
    #[arg(long, default_value_t = 500)]
    testing_class_size: usize,

    /// Learning rate (eta value)
    #[arg(short, long, default_value_t = 3.0)]
    learning_rate: f64,

    /// Number of training cycles before update
    #[arg(short, long, default_value_t = 10)]
    batches: usize,

    /// Number of passes through the entire training set
    #[arg(short, long, default_value_t = 30)]
    epochs: usize,
}

fn main() {
    let args = Args::parse();

    let mut model = RCN::new(
        args.num_classes,
        &[
            RCNLayer::Convolve2D(Padding::Same),
            RCNLayer::Pool2D(Pooling::Max),
            RCNLayer::Convolve2D(Padding::Same),
            RCNLayer::Pool2D(Pooling::Max),
        ],
        &[30],
        &args.training_path,
        &args.testing_path,
    );
    model
        .train(
            args.batches,
            args.epochs,
            args.learning_rate,
            args.training_class_size,
            args.testing_class_size,
        )
        .unwrap();
}
