pub mod utils;

mod errors;

use errors::InvalidGrayscaleImageError;
use image::{io::Reader as ImageReader, ImageError};
use image::{DynamicImage, GenericImageView};
use nalgebra::{DMatrix, DVector};
use rand::seq::SliceRandom;
use rand::{distributions::Uniform, Rng};
use rand_distr::StandardNormal;
use std::f64::consts::E;
use std::{fs, path::PathBuf};
use utils::kernel::{Padding, Pooling, SeparableOperator};

use crate::utils::kernel::{Convolve2D, Pool2D};

/// Rust Convolutional Neural Network (RCN)
pub struct RCN<'a> {
    classes: usize,
    convpool_cfg: &'a [RCNLayer],
    feedforward_cfg: &'a [usize],
    layer_weights: Vec<DMatrix<f64>>,
    layer_bias: Vec<DVector<f64>>,

    training_path: &'a str,
    testing_path: &'a str,
}

/// Characteristics of a layer
pub enum RCNLayer {
    Convolve2D(Padding),
    Pool2D(Pooling),
}

/// List of operators that can be iterated through for convolutions
const SEP_OPS: [SeparableOperator; 4] = [
    SeparableOperator::Top,
    SeparableOperator::Left,
    SeparableOperator::Right,
    SeparableOperator::Bottom,
];

/// The input data for the feedforward portion of the network and the expected output
type InputSet = (DVector<f64>, DVector<f64>);

impl<'a> RCN<'a> {
    /// Create new RCN instance
    ///
    /// # Arguments
    /// * `classes` - The number of final classifications to be made
    /// * `layer_cfg` - A collection of layer configurations
    ///
    pub fn new(
        classes: usize,
        convpool_cfg: &'a [RCNLayer],
        feedforward_cfg: &'a [usize],
        training_path: &'a str,
        testing_path: &'a str,
    ) -> Self {
        RCN {
            classes,
            convpool_cfg,
            feedforward_cfg,
            layer_weights: Vec::new(),
            layer_bias: Vec::new(),
            training_path,
            testing_path,
        }
    }

    /// Classify a given input
    ///
    /// # Arguments
    /// * `x` - The input vector
    ///
    fn classify(&self, x: &DVector<f64>) -> DVector<f64> {
        let mut a: DVector<f64> = x.clone();
        for (b, w) in self.layer_bias.iter().zip(self.layer_weights.iter()) {
            a = sigmoid(&(w * a + b));
        }
        a
    }

    /// Train the model
    ///
    /// # Arguments
    /// * `batch_size` - The number of samples processed before the model is updated
    /// * `epochs` - The number of total passes through the training set
    /// * `eta` - The learning rate
    /// * `class_size_limit` - A limiter on the number of samples to use per class
    ///
    pub fn train(
        &mut self,
        batch_size: usize,
        epochs: usize,
        eta: f64,
        class_size_limit: usize,
    ) -> Result<(), ImageError> {
        let mut training_set: Vec<InputSet> = self.load_data(self.training_path, class_size_limit);
        let mut testing_set: Vec<InputSet> = self.load_data(self.testing_path, class_size_limit);
        self.load_weights_and_bias(training_set[0].0.len());

        println!("{}", self.layer_weights[1]);
        println!("{}", self.layer_bias[0]);

        let (mean, sd) = self.get_scales(&training_set);
        for v in &mut training_set {
            for r in 0..v.0.nrows() {
                let d = (v.0[r] - mean) / sd;
                v.0[r] = if d >= 0_f64 { d } else { 0_f64 };
            }
        }

        let (mean, sd) = self.get_scales(&testing_set);
        for v in &mut testing_set {
            for r in 0..v.0.nrows() {
                let d = (v.0[r] - mean) / sd;
                v.0[r] = if d >= 0_f64 { d } else { 0_f64 };
            }
        }

        // Highest level loop is the epoch loop, since all inner code will be working through the entire dataset
        for e in 0..epochs {
            // Generate batch list
            training_set.shuffle(&mut rand::thread_rng());
            for batch in training_set.chunks_exact(batch_size) {
                self.train_batch(batch, eta);
            }

            println!("{}", self.layer_weights[1]);
            println!("{}", self.layer_bias[0]);

            // Run through test data to show network change
            let mut accept = 0;
            for (test, expectation) in &testing_set {
                let res = self.classify(test);
                let res = res.map(|v| if v == res.max() { 1_f64 } else { 0_f64 });
                accept += if &res == expectation { 1 } else { 0 };
            }
            println!("Epoch {}: {}/{}", e, accept, testing_set.len());
        }
        Ok(())
    }

    /// Train through a batch and then update the parameters
    ///
    /// # Arguments
    /// * `batch` - The input batch which contains an input vector and the
    /// * `eta` - The learning rate
    ///
    fn train_batch(&mut self, batch: &[InputSet], eta: f64) {
        let mut del_w: Vec<DMatrix<f64>> = self
            .layer_weights
            .iter()
            .map(|w| DMatrix::zeros(w.shape().0, w.shape().1))
            .collect();
        let mut del_b: Vec<DVector<f64>> = self
            .layer_bias
            .iter()
            .map(|b| DVector::zeros(b.nrows()))
            .collect();

        for (x, y) in batch {
            let (delta_del_b, delta_del_w) = self.backprop(x, y);
            del_b = delta_del_b
                .iter()
                .zip(del_b.iter())
                .map(|(db, b)| db + b)
                .collect();
            del_w = delta_del_w
                .iter()
                .zip(del_w.iter())
                .map(|(dw, w)| dw + w)
                .collect();
        }

        self.layer_weights = self
            .layer_weights
            .iter()
            .zip(del_w.iter())
            .map(|(lw, w)| lw - (eta / batch.len() as f64) * w)
            .collect();

        self.layer_bias = self
            .layer_bias
            .iter()
            .zip(del_b.iter())
            .map(|(lb, b)| lb - (eta / batch.len() as f64) * b)
            .collect();
    }

    /// Generate scale values for input values
    ///
    /// # Arguments
    /// * `iv` - The input vector to parse through
    ///
    fn get_scales(&self, iv: &Vec<InputSet>) -> (f64, f64) {
        let mut mean = 0_f64;
        let mut sd = 0_f64;
        let N = iv[0].0.len() as f64 * iv.len() as f64;

        for v in iv {
            for r in 0..v.0.nrows() {
                mean += v.0[r];
            }
        }
        mean /= N;

        for v in iv {
            for r in 0..v.0.nrows() {
                sd += f64::powi(v.0[r] - mean, 2);
            }
        }
        sd = f64::sqrt(sd / N);

        (mean, sd)
    }

    /// Return gradient tensors for del_w and del_b
    ///
    /// # Arguments
    /// * `x` - The input vector
    /// * `y` - The expected output
    ///
    fn backprop(
        &self,
        x: &DVector<f64>,
        y: &DVector<f64>,
    ) -> (Vec<DVector<f64>>, Vec<DMatrix<f64>>) {
        let mut del_w: Vec<DMatrix<f64>> = self
            .layer_weights
            .iter()
            .map(|w| DMatrix::zeros(w.shape().0, w.shape().1))
            .collect();
        let mut del_b: Vec<DVector<f64>> = self
            .layer_bias
            .iter()
            .map(|b| DVector::zeros(b.nrows()))
            .collect();

        // List of activations and weighted inputs (zs)
        let mut curr_activation: DVector<f64> = x.clone();
        let mut activations: Vec<DVector<f64>> = vec![x.clone()];
        let mut zs: Vec<DVector<f64>> = Vec::new();

        // println!("Initial Activation:\n{}", curr_activation);

        for (b, w) in self.layer_bias.iter().zip(self.layer_weights.iter()) {
            let z = w * curr_activation + b;
            zs.push(z.clone());
            curr_activation = sigmoid(&z);
            activations.push(curr_activation.clone());
            // println!("Mid-feed Activation:\n{}", curr_activation);
        }

        // println!("Final Activation:\n{}", activations[activations.len() - 1]);

        let db_end = del_b.len() - 1;
        let dw_end = del_w.len() - 1;
        let weight_end = self.layer_weights.len() - 1;
        let activ_end = activations.len() - 1;
        let zs_end = zs.len() - 1;

        let mut delta = self
            .cost_derivative(&activations[activ_end], y)
            .component_mul(&sigmoid_prime(&zs[zs_end]));

        // Bias error is equivalent to delta for all layers
        del_b[db_end] = delta.clone();
        del_w[dw_end] = delta.clone() * activations[activ_end - 1].transpose();

        for l in 1..self.feedforward_cfg.len() + 1 {
            let sp = sigmoid_prime(&zs[zs_end - l]);
            delta = (self.layer_weights[weight_end - l + 1].transpose() * delta).component_mul(&sp);
            del_b[db_end - l] = delta.clone();
            del_w[dw_end - l] = delta.clone() * activations[activ_end - l - 1].transpose();
        }

        // for w in &del_w[1..] {
        //     println!("{}", w);
        // }

        // wait();

        (del_b, del_w)
    }

    fn flatten_feature_set(&self, m: &DMatrix<f64>) -> DVector<f64> {
        let mut feature_set: Vec<DMatrix<f64>> = Vec::new();
        for layer in self.convpool_cfg {
            match layer {
                RCNLayer::Convolve2D(p) => {
                    // build feature set from current matrix if feature_set is not already populated
                    if !feature_set.is_empty() {
                        // preserve current length and iterate over those only
                        let curr_len = feature_set.len();
                        for i in 0..curr_len {
                            let mut it = SEP_OPS.iter().peekable();

                            // if feature set is populated, change current matrix and extend any other matrices
                            while let Some(op) = it.next() {
                                if it.peek().is_none() {
                                    feature_set[i] = feature_set[i].convolve_2d_separated(*op, p)
                                } else {
                                    feature_set.push(feature_set[i].convolve_2d_separated(*op, p));
                                }
                            }
                        }
                    } else {
                        feature_set.extend(SEP_OPS.map(|op| m.convolve_2d_separated(op, p)));
                    }
                }
                RCNLayer::Pool2D(p) => {
                    for feature in &mut feature_set {
                        *feature = feature.pool_2d(&Padding::Same, p);
                    }
                }
            }
        }

        DVector::from(
            feature_set
                .iter()
                .flat_map(|m| m.into_iter().copied().collect::<Vec<f64>>())
                .collect::<Vec<_>>(),
        )
    }

    /// *Internal Function*
    ///
    /// Take a path for data (training or testing/validating) and return a vector of the matrices
    /// and their respective class names.
    ///
    /// # Arguments
    /// * `path` - The path to the data set
    /// * `class_size_limit` - The absolute limit of data that should be populated per class
    ///
    fn load_data(&self, path: &str, class_size_limit: usize) -> Vec<InputSet> {
        let classes: Vec<PathBuf> = fs::read_dir(path)
            .unwrap()
            .map(|f| f.unwrap().path())
            .collect();

        let mut dset: Vec<InputSet> = Vec::new();
        for (i, class) in classes.iter().enumerate() {
            let mut paths: Vec<PathBuf> = fs::read_dir(&class)
                .unwrap()
                .map(|f| f.unwrap().path())
                .collect();

            for _ in 0..class_size_limit {
                let idx = rand::thread_rng().gen_range(0..paths.len());
                let img = ImageReader::open(paths.remove(idx))
                    .unwrap()
                    .decode()
                    .unwrap()
                    .grayscale();
                dset.push((
                    self.flatten_feature_set(&get_pixel_matrix(&img).unwrap()),
                    get_expected_vec(i, classes.len()),
                ));
            }
        }

        dset
    }

    /// *Internal Function*
    ///
    /// Scale the data to be between [0, 1]
    ///
    /// # Arguments
    /// * `v` - The input vector to scale
    ///
    fn scale(&self, v: &mut DVector<f64>) {
        for x in v.iter_mut() {}
    }

    /// *Internal Function*
    ///
    /// Generate the weight matrices based on a given input size for the classes.
    /// Assume that all classes will have the same size inputs.
    ///
    /// # Arguments
    /// * `l` - The length of an input sample, used to calculate the total input layer size
    ///
    fn load_weights_and_bias(&mut self, l: usize) {
        self.layer_weights = Vec::with_capacity(self.feedforward_cfg.len() + 1);
        self.layer_bias = Vec::with_capacity(self.feedforward_cfg.len() + 1);

        let (mut c, mut p) = (0, 0);
        for layer in self.convpool_cfg {
            match layer {
                RCNLayer::Convolve2D(_) => {
                    c += 1;
                }
                RCNLayer::Pool2D(_) => {
                    p += 2;
                }
            }
        }

        // a & b == weight matrix shape (b rows and a columns)
        // 4^c * 1/(2^p) * f_w * f_h = input layer len (initial a)
        let mut a = usize::pow(4, c) / usize::pow(2, p) * l;
        let mut b = self.feedforward_cfg[0];
        for i in 0..self.layer_weights.capacity() {
            self.layer_weights.push(get_he_weight_matrix(a, b));
            self.layer_bias.push(get_he_bias_vector(b));
            (a, b) = (
                b,
                if i + 1 < self.feedforward_cfg.len() {
                    self.feedforward_cfg[i + 1]
                } else {
                    self.classes
                },
            );
        }
    }

    /// Calculate the cost derivative for a given output and expected output
    ///
    /// # Arguments
    /// * `output` - The output activations
    /// * `expected` - The expected output
    ///
    fn cost_derivative(&self, output: &DVector<f64>, expected: &DVector<f64>) -> DVector<f64> {
        output - expected
    }
}

/// Convert a class identifier to a DVector that represents the expected output from the network
///
/// # Arguments
/// * `class_idx` - The value in the vector to set to 1
/// * `classes` - The number of classes, or the size of the vector
///
fn get_expected_vec(class_idx: usize, classes: usize) -> DVector<f64> {
    DVector::from_iterator(
        classes,
        (0..classes).map(|i| if i == class_idx { 1_f64 } else { 0_f64 }),
    )
}

/// Log information of the current image.
pub fn __log_image_info(path: &str, image: &DynamicImage) {
    println!("Image Path: {}", path);
    println!("Dimensions: {:?}", image.dimensions());
    println!("Pixel Count: {}", image.pixels().count());
    println!("Byte Count: {}", image.as_bytes().len());
}

/// Generate a matrix of pixels from a provided image. Every pixel in the pixel iterator is a tuple struct of type
/// Luma that contains a single element array. The first value can be obtained immediately using .0 and then index
/// into the single element.
///
/// Can accept either Luma8 or LumaA8. Alpha channels are ignored.
///
/// # Arguments
/// * `image` - A dynamic image reference, expected to be in grayscale.
///
pub fn get_pixel_matrix(image: &DynamicImage) -> Result<DMatrix<f64>, InvalidGrayscaleImageError> {
    match image {
        DynamicImage::ImageLuma8(gray_image) => Ok(DMatrix::from_row_iterator(
            gray_image.dimensions().1 as usize,
            gray_image.dimensions().0 as usize,
            gray_image.pixels().map(|p| p.0[0] as f64),
        )),
        DynamicImage::ImageLumaA8(gray_image) => Ok(DMatrix::from_row_iterator(
            gray_image.dimensions().1 as usize,
            gray_image.dimensions().0 as usize,
            gray_image.pixels().map(|p| p.0[0] as f64),
        )),
        _ => Err(InvalidGrayscaleImageError),
    }
}

/// Sigmoid activation function
///
/// # Arguments
/// * `v` - The vector to apply the sigmoid function to
///
fn sigmoid(v: &DVector<f64>) -> DVector<f64> {
    DVector::from_iterator(
        v.nrows(),
        v.iter().map(|x| 1_f64 / (1_f64 + f64::powf(E, -*x))),
    )
}

/// Derivative of sigmoid function
///
/// # Arguments
/// * `v` - The vector to apply the derived sigmoid function to
///
fn sigmoid_prime(v: &DVector<f64>) -> DVector<f64> {
    sigmoid(v).component_mul(&sigmoid(v).map(|x| 1_f64 - x))
}

/// Generate set of weights using Xavier initializing.
///
/// # Arguments
/// * `input_size` - The size of the input layer
/// * `output_size` - The size of the output layer
///
fn get_xavier_weight_matrix(input_size: usize, output_size: usize) -> DMatrix<f64> {
    // rows, columns
    let dims = (output_size, input_size + 1);

    let (upper, lower) = (
        1_f64 / (input_size as f64).sqrt(),
        -1_f64 / (input_size as f64).sqrt(),
    );

    DMatrix::from_iterator(
        dims.0,
        dims.1,
        rand::thread_rng()
            .sample_iter(Uniform::new_inclusive(lower, upper))
            .take(dims.0 * dims.1),
    )
}

/// Generate set of weights using He initialization
///
/// # Arguments
/// * `input_size` - The size of the input layer
/// * `output_size` - The size of the output layer
///
fn get_he_weight_matrix(input_size: usize, output_size: usize) -> DMatrix<f64> {
    // rows, columns
    let dims = (output_size, input_size);

    let std: f64 = (2_f64 / input_size as f64).sqrt();

    DMatrix::from_iterator(
        dims.0,
        dims.1,
        rand::thread_rng()
            .sample_iter(StandardNormal)
            .take(dims.0 * dims.1), // .map(|n: f64| n * std),
    )
}

/// Generate set of biases using He initialization
///
/// # Arguments
/// * `neurons` - The number of neurons to build a bias for
///
fn get_he_bias_vector(neurons: usize) -> DVector<f64> {
    DVector::from_iterator(
        neurons,
        rand::thread_rng().sample_iter(StandardNormal).take(neurons),
    )
}

fn wait() {
    use std::io::stdin;
    let mut s = String::new();
    stdin().read_line(&mut s).unwrap();
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    /// Test obtaining weight values
    fn xavier_init() {
        let (input_size, output_size) = (100, 32);
        assert_eq!(
            get_xavier_weight_matrix(input_size, output_size).len(),
            output_size * (input_size + 1)
        );
    }

    #[test]
    /// Print sample matrix from init_weight
    fn print_xavier_init() {
        let (input_size, output_size) = (5, 10);
        println!("{}", get_xavier_weight_matrix(input_size, output_size));
    }

    #[test]
    /// Verify no value in the weight matrix is outside the distribution range
    fn check_weight_in_range() {
        let (input_size, output_size) = (30, 300);
        let m = get_xavier_weight_matrix(input_size, output_size);

        let (upper, lower) = (
            1_f64 / (input_size as f64).sqrt(),
            -1_f64 / (input_size as f64).sqrt(),
        );

        for x in m.into_iter() {
            assert!(*x >= lower && *x <= upper);
        }
    }

    #[test]
    /// Test obtaining weight values
    fn he_init() {
        let (input_size, output_size) = (100, 32);
        assert_eq!(
            get_he_weight_matrix(input_size, output_size).len(),
            output_size * input_size
        );
    }

    #[test]
    /// Print sample matrix from init_weight
    fn print_he_init() {
        let (input_size, output_size) = (5, 10);
        println!("{}", get_he_weight_matrix(input_size, output_size));
    }

    #[test]
    /// Creating and configuring RCN
    fn rcn_init() {
        let model = RCN::new(
            10,
            &[
                RCNLayer::Convolve2D(Padding::Same),
                RCNLayer::Pool2D(Pooling::Max),
                RCNLayer::Convolve2D(Padding::Same),
                RCNLayer::Pool2D(Pooling::Max),
            ],
            &[10, 10],
            "images\\mnist_png\\train",
            "images\\mnist_png\\valid",
        );

        assert_eq!(model.classes, 10);
    }

    #[test]
    fn training() {
        let mut model = RCN::new(
            10,
            &[
                RCNLayer::Convolve2D(Padding::Same),
                RCNLayer::Pool2D(Pooling::Max),
                RCNLayer::Convolve2D(Padding::Same),
                RCNLayer::Pool2D(Pooling::Max),
            ],
            &[10, 10],
            "images\\mnist_png\\train",
            "images\\mnist_png\\valid",
        );

        model.train(10, 10, 3_f64, 50).unwrap();
    }
}
