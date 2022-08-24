// Current nalgebra library supports vector (as in column matrix) convolutions but not 2D matrix convolutions.
// This code is an extension on the nalgebra library to allow this.

// Note: cannot define an implementation outside of an external crate for structs in said crate. Therefore, use
// traits instead to force an external type to be bound by some trait.

use nalgebra::{matrix, DMatrix, Dim, Dynamic, Matrix, Matrix3, Scalar, Storage};
use num::Zero;
use std::ops::{AddAssign, Mul};

pub const TOP_SOBEL: Matrix3<i16> = matrix![1, 2, 1; 0, 0, 0; -1, -2, -1];
pub const BOTTOM_SOBEL: Matrix3<i16> = matrix![-1, -2, -1; 0, 0, 0; 1, 2, 1];
pub const LEFT_SOBEL: Matrix3<i16> = matrix![1, 0, -1; 2, 0, -2; 1, 0, -1];
pub const RIGHT_SOBEL: Matrix3<i16> = matrix![-1, 0, 1; -2, 0, 2; -1, 0, 1];

pub trait Convolve2D<N, R1, C1, S1>
where
    N: Scalar + Zero + AddAssign + Mul<Output = N> + Copy,
    R1: Dim,
    C1: Dim,
    S1: Storage<N, R1, C1>,
{
    /// Returns the convolution of a target 2D Matrix and a kernel.
    ///
    /// # Arguments
    /// * `kernel` - A Matrix with size > 0
    ///
    /// # Errors
    /// Kernel dimensions must be less than or equal to target matrix's dimensions.
    ///
    fn convolve_2d<R2, C2, S2>(&self, kernel: &Matrix<N, R2, C2, S2>) -> DMatrix<N>
    where
        R2: Dim,
        C2: Dim,
        S2: Storage<N, R2, C2>;
}

impl<N, R1, C1, S1> Convolve2D<N, R1, C1, S1> for Matrix<N, R1, C1, S1>
where
    N: Scalar + Zero + AddAssign + Mul<Output = N> + Copy,
    R1: Dim,
    C1: Dim,
    S1: Storage<N, R1, C1>,
{
    // Credit to GitHub user guissalustiano on his nalgebra pull request (#855) on how the Storage trait is used
    fn convolve_2d<R2, C2, S2>(&self, kernel: &Matrix<N, R2, C2, S2>) -> DMatrix<N>
    where
        R2: Dim,
        C2: Dim,
        S2: Storage<N, R2, C2>,
    {
        let matrix_shape = self.shape();
        let kernel_shape = kernel.shape();

        if kernel_shape == (0, 0)
            || kernel_shape.0 > matrix_shape.0
            || kernel_shape.1 > matrix_shape.1
        {
            panic!("convolve_2d expects 'self.shape() >= kernel_shape() > 0', received {:?} and {:?} respectively.", matrix_shape, kernel_shape);
        }

        let conv_shape = (
            matrix_shape.0 - kernel_shape.0 + 1,
            matrix_shape.1 - kernel_shape.1 + 1,
        );

        let mut conv = DMatrix::zeros_generic(
            Dynamic::from_usize(conv_shape.0),
            Dynamic::from_usize(conv_shape.1),
        );

        for c_y in 0..conv_shape.0 {
            for c_x in 0..conv_shape.1 {
                for k_y in 0..kernel_shape.0 {
                    for k_x in 0..kernel_shape.1 {
                        conv[(c_x, c_y)] += self[(c_x + k_x, c_y + k_y)] * kernel[(k_x, k_y)];
                    }
                }
            }
        }

        conv
    }
}

#[cfg(test)]
mod tests {

    use std::collections::HashMap;

    use super::*;
    use crate::get_pixel_matrix;
    use image::{io::Reader as ImageReader, ImageError};
    use nalgebra::VecStorage;

    #[test]
    /// View what a convolved image of the number 4 (grayscale) would look like.
    fn convolve_2d() -> Result<(), ImageError> {
        let img = ImageReader::open("images\\mnist_png\\train\\4\\2.png")?.decode()?;
        let matrix = get_pixel_matrix(&img).unwrap();

        println!("{}", matrix);

        let normalize = |i: &i16| -> u8 {
            if *i > 255 {
                255 as u8
            } else if *i < 0 {
                0 as u8
            } else {
                *i as u8
            }
        };

        let save =
            |name: &str,
             matrix: Matrix<i16, Dynamic, Dynamic, VecStorage<i16, Dynamic, Dynamic>>| {
                let buf: Vec<u8> = matrix.transpose().iter().map(|i| normalize(i)).collect();
                image::save_buffer(
                    name,
                    &buf,
                    matrix.shape().0 as u32,
                    matrix.shape().1 as u32,
                    image::ColorType::L8,
                )
                .unwrap();
            };

        let top_conv_result = matrix.convolve_2d(&TOP_SOBEL);
        let bottom_conv_result = matrix.convolve_2d(&BOTTOM_SOBEL);
        let left_conv_result = matrix.convolve_2d(&LEFT_SOBEL);
        let right_conv_result = matrix.convolve_2d(&RIGHT_SOBEL);

        save("test\\top.png", top_conv_result);
        save("test\\bottom.png", bottom_conv_result);
        save("test\\left.png", left_conv_result);
        save("test\\right.png", right_conv_result);

        Ok(())
    }

    #[test]
    fn print_sobels() {
        let sobels = HashMap::from([
            ("Top", TOP_SOBEL),
            ("Bottom", BOTTOM_SOBEL),
            ("Left", LEFT_SOBEL),
            ("Right", RIGHT_SOBEL),
        ]);

        for (k, v) in sobels {
            println!("{}", k);

            for r in 0..3 {
                for c in 0..3 {
                    print!("{:>4}", v[(r, c)]);
                }
                println!();
            }
            println!();
        }
    }
}
