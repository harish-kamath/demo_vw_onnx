# VW + ONNX CSOAA Demo
A demo of how to use a VW CSOAA model in onnxruntime.

### Create VW CSOAA Model, export to ONNX format
- Follow the guide [here](https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Cost-Sensitive-One-Against-All-(csoaa)-multi-class-example) to create a VW CSOAA model on some training set - use the argument `--invert_hash invhash.txt` to create the model in readable format with hash values for features, saved as `invhash.txt`. 
- Copy `invhash.txt` into this repo. Now, run `python create_model.py` to create a `model.onnx`.
- Change the variable `MODEL_URI` in `app/main.cc` to point to  this onnx file.

### Build and run executable
For dependencies / environment set up, see https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Dependencies

    git submodule update --init
    ./lib/onnxruntime/build.sh --config RelWithDebInfo --build_shared_lib --parallel
    mkdir build
    cd build
    cmake ..
    make

Build has been tested on Ubuntu 16.04
Executable located at `build/app/vw_onnx_csoaa_demo`. This executable takes only one argument: The example you want to run inference on in vw format.

Example use: `vw_onnx_csoaa_demo "1:1.0 a1_expect_1| a"`
