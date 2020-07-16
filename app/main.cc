#include <iostream>
#include <fstream>
#include "vw.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/graph/constants.h"

struct Input {
  const char* name;
  std::vector<int64_t> dims;
  std::vector<uint32_t> values_int;
  std::vector<float> values_float;
};

struct OrtTensorDimensions : std::vector<int64_t> {
  OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue* value) {
    OrtTensorTypeAndShapeInfo* info = ort.GetTensorTypeAndShape(value);
    std::vector<int64_t>::operator=(ort.GetTensorShape(info));
    ort.ReleaseTensorTypeAndShapeInfo(info);
  }
};

typedef const char* PATH_TYPE;
#define TSTR(X) (X)
static constexpr PATH_TYPE MODEL_URI = TSTR("../model.onnx"); // CHANGE TO POINT TO ONNX MODEL

int main(int argc, char** argv)
{
    //ORT
    Ort::Env env_= Ort::Env(ORT_LOGGING_LEVEL_INFO, "Default");
    
    
// Inference
    Ort::SessionOptions session_options;
    Ort::Session session(env_, MODEL_URI, session_options);
    Ort::AllocatorWithDefaultOptions allocator;
    std::string classCount = session.GetModelMetadata().LookupCustomMetadataMap("class_count", allocator);
    
    std::string line = std::string(argv[1]);
    
    std::string initializer = std::string(" -b 32") + std::string(" --csoaa ") + classCount + std::string(" --quiet");
    auto vw = VW::initialize(initializer);
    example* ex = VW::read_example(*vw, line);
    
    std::vector<Input> inputs(2);
      auto input = inputs.begin();
    input->name = "Features";
    input->values_int = {};
    
    for (features& fs : *ex) {
        for (features::iterator& f : fs)
        {
            input->values_int.push_back(f.index());
        }
    }
    VW::finish_example(*vw, *ex);
    
    input->dims = {static_cast<int>(input->values_int.size())};

    
      input = std::next(input, 1);
      input->name = "Valid_Classes";
      input->dims = {3};
        input->values_float = {0.0f,1.0f,0.0f};
      
      const char* output_name = "Outputs";
    
    
    
    
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    std::vector<Ort::Value> input_tensors;
    std::vector<const char*> input_names;

    for (size_t i = 0; i < inputs.size(); i++) {
      input_names.emplace_back(inputs[i].name);
        if(i == 0){
      input_tensors.emplace_back(Ort::Value::CreateTensor<uint32_t>(memory_info, const_cast<uint32_t*>(inputs[i].values_int.data()), inputs[i].values_int.size(), inputs[i].dims.data(), inputs[i].dims.size()));
        }
        else{
        input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(inputs[i].values_float.data()), inputs[i].values_float.size(), inputs[i].dims.data(), inputs[i].dims.size()));
        }
    }

    std::vector<Ort::Value> ort_outputs;
    ort_outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), &output_name, 1);
    int64_t* f = ort_outputs[0].GetTensorMutableData<int64_t>();
    
    std::cout << "Model Output: " << *f << "\n";
    return 0;
    
    
/*
    std::fstream data_file(argv[1]);
    std::string line;
    while (std::getline(data_file, line))
    {
        // assumes single-line examples; multi-line (e.g. for CB) is more complex
        example* ex = VW::read_example(*vw, line);

        vw->learn(*ex);

        VW::finish_example(*vw, *ex);

        // note that this is a manual driver. If you want to use vw in the same mode as the executable
        // use this code instead (and add appropriate -i, -f, -d, etc. options to the args passed to VW::initialize)
        
        VW::start_parser(*vw);
        LEARNER::generic_driver(*vw);
        VW::end_parser(*vw);

        VW::sync_stats(*vw);
        
        
    }

    VW::save_predictor(*vw, argv[2]);
*/
    VW::finish(*vw);

    return 0;
}
