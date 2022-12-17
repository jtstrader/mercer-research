use mercer_research::{
    utils::kernel::{Padding, Pooling},
    RCNLayer, RCN,
};

fn main() {
    // expectation: be able to create a CNN with any given number of layers and have those layers
    // stored in some sort of struct. Something maybe like this? ->
    //
    // let model = MyCNNStruct::new(num_convolutional_layers, num_fully_connected_layers, activation_function)
    //
    // of course, this is just a concept and might change, but this should help build the foundations of the library functions

    let mut model = RCN::new(
        10,
        &[
            RCNLayer::Convolve2D(Padding::Same),
            RCNLayer::Pool2D(Pooling::Max),
        ],
        &[30],
        "images\\mnist_png\\train",
        "images\\mnist_png\\valid",
    );

    model.train(10, 50, 3_f64, 100).unwrap();
}
