use actix_cors::Cors;
use actix_web::{get, web, App, HttpServer, Responder};
use base64::engine::{general_purpose, Engine as _};
use log::debug;
use rand::{seq::SliceRandom, Rng};
use rcn::rcn::RCN;
use serde::Serialize;
use std::{fs, io::Cursor, path::PathBuf};

struct RCNState<'a> {
    model: RCN<'a>,
    image_paths: &'static [PathBuf],
}

#[derive(Serialize)]
struct RCNResult {
    output: usize,
    img: String,
}

/// Select random file, classify it, and return the file data and classification to the caller.
#[get("/")]
async fn get_rcn_result(data: web::Data<RCNState<'_>>) -> actix_web::Result<impl Responder> {
    let model = &data.model;
    let paths = &data.image_paths;
    let mut gen = rand::thread_rng();

    let file = &paths[gen.gen_range(0..paths.len())];
    let result = model.classify(&file.to_string_lossy())?;

    let img = image::open(file).unwrap();
    let mut buf: Vec<u8> = Vec::new();
    img.write_to(&mut Cursor::new(&mut buf), image::ImageOutputFormat::Png)
        .unwrap();

    debug!("Request received for image, returning {}", result);

    Ok(web::Json(RCNResult {
        output: result,
        img: general_purpose::STANDARD.encode(buf),
    }))
}

#[get("/health")]
async fn health() -> String {
    "Healthy!".to_owned()
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    std::env::set_var("RUST_LOG", "debug");
    env_logger::init();

    let serialized_model: &'static [u8] = Vec::leak(fs::read("../rcn/rcn.bin")?);
    let image_paths: &'static [PathBuf] = {
        let mut v = Vec::with_capacity(10);
        for sub_dir in fs::read_dir("images")?.map(|d| d.unwrap().path()) {
            fs::read_dir(sub_dir)?.for_each(|p| v.push(p.unwrap().path()));
        }
        v.shuffle(&mut rand::thread_rng());
        Vec::leak(v)
    };

    HttpServer::new(move || {
        App::new()
            .wrap(Cors::permissive())
            .app_data(web::Data::new(RCNState {
                model: bincode::deserialize(&serialized_model[..])
                    .expect("model serialized with correct format"),
                image_paths,
            }))
            .service(get_rcn_result)
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}
