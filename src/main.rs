use image::{io::Reader as ImageReader, ImageError};
use mercer_research::{get_pixel_matrix, log_image_info};

fn main() -> Result<(), ImageError> {
    // Test loading an image of the number 4 from the mnist training set
    test_image_get("images\\mnist_png\\train\\4\\2.png")?;

    Ok(())
}

fn test_image_get(image_path: &str) -> Result<(), ImageError> {
    let base_img = ImageReader::open(image_path)?.decode()?;

    log_image_info(image_path, &base_img);
    println!();

    let pixel_matrix = match get_pixel_matrix(&base_img) {
        Ok(mx) => mx,
        Err(e) => {
            panic!("{}", e);
        }
    };

    for y in 0..28 {
        for x in 0..28 {
            print!("{:<3} ", pixel_matrix[(x, y)]);
        }
        println!();
    }
    println!();
    Ok(())
}
