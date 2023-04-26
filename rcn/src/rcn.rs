use crate::utils::kernel::{Convolve2D, Padding, Pool2D, Pooling, SeparableOperator};
use image::{io::Reader as ImageReader, ImageError};
use nalgebra::{DMatrix, DVector};
use rand::seq::SliceRandom;
use rand::Rng;
use rand_distr::StandardNormal;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f64::consts::E;
use std::sync::{Arc, Mutex};
use std::{fs, path::PathBuf};

#[derive(Serialize, Deserialize)]
/// Rust Convolutional Neural Network (RCN)
pub struct RCN<'a> {
    classes: usize,
    convpool_cfg: Vec<RCNLayer>,
    feedforward_cfg: Vec<usize>,
    layer_weights: Vec<Weights>,
    layer_bias: Vec<Bias>,
    scale_set: (f64, f64),

    training_path: &'a str,
    testing_path: &'a str,

    data_fmt: DataFormat,
}

/// Weights tuple struct wrapped for serializing/deserializing.
pub struct Weights(pub DMatrix<f64>);

/// Biases tuple struct wrapped for serializing/deserializing.
pub struct Bias(pub DVector<f64>);

#[derive(Serialize, Deserialize)]
/// Characteristics of a layer
pub enum RCNLayer {
    Convolve2D(Padding),
    Pool2D(Pooling),
}

#[derive(Serialize, Deserialize)]
pub enum DataFormat {
    DirWalk,
    Csv,
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

/// Unformated input data.
type IntermediateInputSet = (DMatrix<f64>, DVector<f64>);

impl<'a> RCN<'a> {
    /// Create new RCN instance
    ///
    /// # Arguments
    /// * `classes` - The number of final classifications to be made
    /// * `layer_cfg` - A collection of layer configurations
    ///
    pub fn new(
        classes: usize,
        convpool_cfg: Vec<RCNLayer>,
        feedforward_cfg: Vec<usize>,
        training_path: &'a str,
        testing_path: &'a str,
        data_fmt: DataFormat,
    ) -> Self {
        RCN {
            classes,
            convpool_cfg,
            feedforward_cfg,
            layer_weights: Vec::new(),
            layer_bias: Vec::new(),
            scale_set: (1_f64, 1_f64),
            training_path,
            testing_path,
            data_fmt,
        }
    }

    /// Classify a given image and return the respective class index.
    ///
    /// # Arguments
    /// * `img_path` - A path to an image to classify
    ///
    pub fn classify(&self, img_path: &str) -> Result<usize, Box<dyn std::error::Error>> {
        let img = ImageReader::open(img_path)?.decode()?.grayscale();
        let mut input_vector = self.flatten_feature_set(&crate::get_pixel_matrix(&img)?);

        for r in 0..input_vector.nrows() {
            let d = (input_vector[r] - self.scale_set.0) / self.scale_set.1;
            input_vector[r] = if d >= 0_f64 { d } else { 0_f64 };
        }

        let result_vector = self.classify_test(&input_vector);
        Ok(result_vector
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .expect("network cannot have 0 classes"))
    }

    /// Classify a given input for testing the model
    ///
    /// # Arguments
    /// * `x` - The input vector
    ///
    fn classify_test(&self, x: &DVector<f64>) -> DVector<f64> {
        let mut a: DVector<f64> = x.clone();
        for (b, w) in self
            .layer_bias
            .iter()
            .zip(self.layer_weights.iter())
            .map(|(b, w)| (&b.0, &w.0))
        {
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
        training_class_size_limit: usize,
        testing_class_size_limit: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut training_set: Vec<InputSet> =
            self.load_data(self.training_path, training_class_size_limit)?;
        let testing_set: Vec<InputSet> =
            self.load_data(self.testing_path, testing_class_size_limit)?;

        if self.layer_weights.is_empty() || self.layer_bias.is_empty() {
            self.load_weights_and_bias(training_set[0].0.len());
        }

        // Highest level loop is the epoch loop, since all inner code will be working through the entire dataset
        for e in 0..epochs {
            // Generate batch list
            training_set.shuffle(&mut rand::thread_rng());
            for batch in training_set.chunks_exact(batch_size) {
                self.train_batch(batch, eta);
            }

            // Run through test data to show network change
            let mut accept = 0;
            for (test, expectation) in &testing_set {
                let res = self.classify_test(test);
                let res = res.map(|v| if v == res.max() { 1_f64 } else { 0_f64 });
                accept += i32::from(&res == expectation);
            }
            println!(
                "Epoch {}: {}/{} [{:.2}%]",
                e,
                accept,
                testing_set.len(),
                (accept as f64 / testing_set.len() as f64) * 100_f64
            );
        }
        Ok(())
    }

    /// Train through a batch and then update the parameters
    ///
    /// # Arguments
    /// * `batch` - The input batch which contains an input vector and the
    /// * `eta` - The learning rate
    ///
    #[inline]
    fn train_batch(&mut self, batch: &[InputSet], eta: f64) {
        let del_w: Arc<Mutex<Vec<DMatrix<f64>>>> = Arc::new(Mutex::new(
            self.layer_weights
                .iter()
                .map(|w| DMatrix::zeros(w.0.shape().0, w.0.shape().1))
                .collect(),
        ));
        let del_b: Arc<Mutex<Vec<DVector<f64>>>> = Arc::new(Mutex::new(
            self.layer_bias
                .iter()
                .map(|b| DVector::zeros(b.0.nrows()))
                .collect(),
        ));

        batch.par_iter().for_each(|(x, y)| {
            let (delta_del_b, delta_del_w) = self.backprop(x, y);
            let mut del_ub = del_b.lock().unwrap();
            let mut del_uw = del_w.lock().unwrap();

            *del_ub = delta_del_b
                .iter()
                .zip(del_ub.iter())
                .map(|(db, b)| db + b)
                .collect();
            *del_uw = delta_del_w
                .iter()
                .zip(del_uw.iter())
                .map(|(dw, w)| dw + w)
                .collect();
        });

        let del_w = Arc::try_unwrap(del_w).unwrap().into_inner().unwrap();
        let del_b = Arc::try_unwrap(del_b).unwrap().into_inner().unwrap();

        self.layer_weights = self
            .layer_weights
            .iter()
            .zip(del_w.iter())
            .map(|(lw, w)| Weights(&lw.0 - (eta / batch.len() as f64) * w))
            .collect();

        self.layer_bias = self
            .layer_bias
            .iter()
            .zip(del_b.iter())
            .map(|(lb, b)| Bias(&lb.0 - (eta / batch.len() as f64) * b))
            .collect();
    }

    /// Generate scale values for input values
    ///
    /// # Arguments
    /// * `iv` - The input vector to parse through
    ///
    fn gen_scales(&mut self, iv: &Vec<InputSet>) {
        let mut mean = 0_f64;
        let mut sd = 0_f64;
        let n = iv[0].0.len() as f64 * iv.len() as f64;

        for v in iv {
            for r in 0..v.0.nrows() {
                mean += v.0[r];
            }
        }
        mean /= n;

        for v in iv {
            for r in 0..v.0.nrows() {
                sd += f64::powi(v.0[r] - mean, 2);
            }
        }
        sd = f64::sqrt(sd / n);

        self.scale_set.0 = mean;
        self.scale_set.1 = sd;
    }

    /// Return gradient tensors for del_w and del_b
    ///
    /// # Arguments
    /// * `x` - The input vector
    /// * `y` - The expected output
    ///
    #[inline]
    fn backprop(
        &self,
        x: &DVector<f64>,
        y: &DVector<f64>,
    ) -> (Vec<DVector<f64>>, Vec<DMatrix<f64>>) {
        let mut del_w: Vec<DMatrix<f64>> = self
            .layer_weights
            .iter()
            .map(|w| DMatrix::zeros(w.0.shape().0, w.0.shape().1))
            .collect();
        let mut del_b: Vec<DVector<f64>> = self
            .layer_bias
            .iter()
            .map(|b| DVector::zeros(b.0.nrows()))
            .collect();

        // List of activations and weighted inputs (zs)
        let mut curr_activation: DVector<f64> = x.clone();
        let mut activations: Vec<DVector<f64>> = vec![x.clone()];
        let mut zs: Vec<DVector<f64>> = Vec::new();

        for (b, w) in self
            .layer_bias
            .iter()
            .zip(self.layer_weights.iter())
            .map(|(b, w)| (&b.0, &w.0))
        {
            let z = w * curr_activation + b;
            zs.push(z.clone());
            curr_activation = sigmoid(&z);
            activations.push(curr_activation.clone());
        }

        let db_end = del_b.len() - 1;
        let dw_end = del_w.len() - 1;
        let weight_end = self.layer_weights.len() - 1;
        let activ_end = activations.len() - 1;
        let zs_end = zs.len() - 1;

        let mut delta = (&activations[activ_end] - y).component_mul(&sigmoid_prime(&zs[zs_end]));

        // Bias error is equivalent to delta for all layers
        del_b[db_end] = delta.clone();
        del_w[dw_end] = delta.clone() * activations[activ_end - 1].transpose();

        for l in 1..self.feedforward_cfg.len() + 1 {
            let sp = sigmoid_prime(&zs[zs_end - l]);
            delta =
                (self.layer_weights[weight_end - l + 1].0.transpose() * delta).component_mul(&sp);
            del_b[db_end - l] = delta.clone();
            del_w[dw_end - l] = &delta * activations[activ_end - l - 1].transpose();
        }

        (del_b, del_w)
    }

    #[inline]
    fn flatten_feature_set(&self, m: &DMatrix<f64>) -> DVector<f64> {
        if self.convpool_cfg.is_empty() {
            return DVector::from(m.iter().copied().collect::<Vec<f64>>());
        }

        let mut feature_set: Vec<DMatrix<f64>> = Vec::new();
        for layer in &self.convpool_cfg {
            match layer {
                RCNLayer::Convolve2D(ref p) => {
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
                RCNLayer::Pool2D(ref p) => {
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
    fn load_data(
        &mut self,
        path: &str,
        class_size_limit: usize,
    ) -> Result<Vec<InputSet>, Box<dyn std::error::Error>> {
        let intermediate_dset = match self.data_fmt {
            DataFormat::DirWalk => from_dir_walk(path, class_size_limit)?,
            DataFormat::Csv => from_csv(path, self.classes)?,
        };

        let mut dset: Vec<InputSet> = intermediate_dset
            .into_iter()
            .map(|(input, output)| (self.flatten_feature_set(&input), output))
            .collect();

        self.gen_scales(&dset);
        for v in &mut dset {
            for r in 0..v.0.nrows() {
                let d = (v.0[r] - self.scale_set.0) / self.scale_set.1;
                v.0[r] = if d >= 0_f64 { d } else { 0_f64 };
            }
        }

        Ok(dset)
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
        for layer in &self.convpool_cfg {
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
            self.layer_weights.push(Weights(get_weight_matrix(a, b)));
            self.layer_bias.push(Bias(get_bias_vector(b)));
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

/// Generate set of weights
///
/// # Arguments
/// * `input_size` - The size of the input layer
/// * `output_size` - The size of the output layer
///
fn get_weight_matrix(input_size: usize, output_size: usize) -> DMatrix<f64> {
    // rows, columns
    let dims = (output_size, input_size);

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
fn get_bias_vector(neurons: usize) -> DVector<f64> {
    DVector::from_iterator(
        neurons,
        rand::thread_rng().sample_iter(StandardNormal).take(neurons),
    )
}

/// Read data from a directory walk and return a relevant input set from it.
fn from_dir_walk(
    path: &str,
    class_size_limit: usize,
) -> Result<Vec<IntermediateInputSet>, ImageError> {
    let mut classes: Vec<_> = fs::read_dir(path)?
        .into_iter()
        .filter_map(|entry| entry.ok())
        .map(|e| e.path())
        .collect();
    classes.sort();

    let mut dset: Vec<IntermediateInputSet> = Vec::new();
    for (i, class) in classes.iter().enumerate() {
        let mut paths: Vec<PathBuf> = fs::read_dir(class)
            .unwrap()
            .map(|f| f.unwrap().path())
            .collect();

        if class_size_limit > paths.len() {
            panic!(
                "provided class_size_limit for {} too large! expected {} <= {}",
                path,
                class_size_limit,
                paths.len()
            );
        }

        let exp_vec = get_expected_vec(i, classes.len());

        for _ in 0..class_size_limit {
            let idx = rand::thread_rng().gen_range(0..paths.len());
            let img = ImageReader::open(paths.remove(idx))?.decode()?.grayscale();
            dset.push((crate::get_pixel_matrix(&img).unwrap(), exp_vec.clone()));
        }
    }
    Ok(dset)
}

pub fn from_csv(
    path: &str,
    classes: usize,
) -> Result<Vec<IntermediateInputSet>, Box<dyn std::error::Error>> {
    const WIDTH: usize = 28;
    const HEIGHT: usize = 28;

    let mut csv_rdr = csv::Reader::from_reader(fs::File::open(path)?);
    let mut dset: Vec<IntermediateInputSet> = Vec::new();
    for record_result in csv_rdr.records() {
        let record = record_result?;
        let mut ri = record.into_iter();
        let class_idx: usize = ri.next().ok_or("empty record")?.parse()?;
        let m: DMatrix<f64> = DMatrix::from_iterator(
            WIDTH,
            HEIGHT,
            ri.map(|pixel| pixel.parse::<f64>().expect("valid pixel information")),
        );
        dset.push((m, get_expected_vec(class_idx, classes)));
    }
    Ok(dset)
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    /// Test obtaining weight values
    fn weight_init() {
        let (input_size, output_size) = (100, 32);
        assert_eq!(
            get_weight_matrix(input_size, output_size).len(),
            output_size * input_size
        );
    }

    #[test]
    /// Print sample matrix from init_weight
    fn print_weight_init() {
        let (input_size, output_size) = (5, 10);
        println!("{}", get_weight_matrix(input_size, output_size));
    }

    #[test]
    /// Creating and configuring RCN
    fn rcn_init() {
        let model = RCN::new(
            10,
            vec![
                RCNLayer::Convolve2D(Padding::Same),
                RCNLayer::Pool2D(Pooling::Max),
                RCNLayer::Convolve2D(Padding::Same),
                RCNLayer::Pool2D(Pooling::Max),
            ],
            vec![10, 10],
            "images/mnist_png/training",
            "images/mnist_png/testing",
            DataFormat::DirWalk,
        );

        assert_eq!(model.classes, 10);
    }

    #[test]
    fn training() {
        let mut model = RCN::new(
            10,
            vec![
                RCNLayer::Convolve2D(Padding::Same),
                RCNLayer::Pool2D(Pooling::Max),
                RCNLayer::Convolve2D(Padding::Same),
                RCNLayer::Pool2D(Pooling::Max),
            ],
            vec![10, 10],
            "images/mnist_png/training",
            "images/mnist_png/testing",
            DataFormat::DirWalk,
        );

        model.train(10, 10, 3_f64, 50, 50).unwrap();
    }
}
