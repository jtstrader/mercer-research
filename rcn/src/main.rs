use clap::Parser;
use rcn::{
    rcn::{DataFormat, RCNLayer, RCN},
    utils::kernel::{Padding, Pooling},
};
use std::{ffi::OsStr, fs, path::Path};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number of classes
    #[arg(short, long, default_value_t = 10)]
    num_classes: usize,

    /// Training directory
    #[arg(long, default_value_t = String::from("images/archive/fashion-mnist_train.csv"))]
    training_path: String,

    /// Testing/validation directory
    #[arg(long, default_value_t = String::from("images/archive/fashion-mnist_test.csv"))]
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

fn main() -> bincode::Result<()> {
    // Load arguments and stored model, if possible.
    let args = Args::parse();
    let serialized_model = fs::read("./rcn.bin").ok();
    let data_fmt = get_data_fmt(&args.training_path).unwrap();

    let mut model: RCN = match serialized_model {
        Some(ref data) => bincode::deserialize(&data[..])?,
        None => RCN::new(
            args.num_classes,
            vec![], // no convolutions
            vec![30],
            &args.training_path,
            &args.testing_path,
            data_fmt,
        ),
    };

    match model.train(
        args.batches,
        args.epochs,
        args.learning_rate,
        args.training_class_size,
        args.testing_class_size,
    ) {
        Ok(()) => {}
        Err(e) => eprintln!("{e}"),
    };

    // Serialize data to file
    fs::write("./rcn.bin", bincode::serialize(&model)?)?;

    Ok(())
}

fn get_data_fmt(path: &str) -> Result<DataFormat, &str> {
    let path = Path::new(path);
    if path.is_dir() {
        return Ok(DataFormat::DirWalk);
    }

    let Some(ext) = path.extension().and_then(OsStr::to_str) else {
        return Err("invalid file name entered");
    };

    match ext {
        "csv" => Ok(DataFormat::Csv),
        _ => Err("invalid file extension"),
    }
}
