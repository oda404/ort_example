
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION

#include <onnxruntime_training_cxx_api.h>
#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <cstdint>
#include <thread>
#include <opencv2/opencv.hpp>
#include <random>
#include <algorithm>

int main()
{
    std::vector<std::string> train_data_paths;
    std::vector<std::string> train_data_labels;

    std::string train_model_path = "../res/mobilenetv2_training.onnx";
    std::string eval_model_path = "../res/mobilenetv2_eval.onnx";
    std::string chk_path = "../res/mobilenetv2.ckpt";
    std::string optim_path = "../res/mobilenetv2_optimizer.onnx";

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ort");
    Ort::SessionOptions opts;
    Ort::CheckpointState chk(Ort::CheckpointState::LoadCheckpoint(chk_path.c_str()));

    Ort::TrainingSession session(opts, chk, train_model_path, eval_model_path, optim_path);

    for (size_t i = 1; i <= 20; ++i)
    {
        std::string n = i < 10 ? "0" + std::to_string(i) : std::to_string(i);

        train_data_paths.push_back("../res/images/cow/cow" + n + ".jpeg");
        train_data_labels.push_back("cow");

        train_data_paths.push_back("../res/images/cat/cat" + n + ".jpeg");
        train_data_labels.push_back("cat");

        train_data_paths.push_back("../res/images/dog/dog" + n + ".jpeg");
        train_data_labels.push_back("dog");

        train_data_paths.push_back("../res/images/elephant/elephant" + n + ".jpeg");
        train_data_labels.push_back("elephant");
    }

    int epochs = 100;
    int batch_size = 1;
    int w = 224;
    int h = 224;
    int channels = 3;

    float *buf = (float *)malloc(224 * 224 * 3 * sizeof(float));
    auto rng = std::default_random_engine{};

    for (size_t i = 0; i < epochs; ++i)
    {
        float loss = 0;
        std::vector<size_t> rand_idx;
        for (size_t k = 0; k < 80; ++k)
            rand_idx.push_back(k);
        std::shuffle(std::begin(rand_idx), std::end(rand_idx), rng);

        for (size_t k = 0; k < 80; ++k)
        {
            std::string filepath = train_data_paths[rand_idx[k]];
            std::string label = train_data_labels[rand_idx[k]];

            cv::Mat img = cv::imread(filepath);
            cv::Mat out;
            cv::resize(img, out, cv::Size(224, 224));

            for (size_t x = 0; x < out.cols; ++x)
            {
                for (size_t y = 0; y < out.rows; ++y)
                {
                    cv::Vec3b color = out.at<cv::Vec3b>(y, x);
                    size_t idx = x * out.rows + y;

                    buf[idx + 224 * 224 * 2] = ((color[0] / 255.f) - 0.406f) / 0.225f;
                    buf[idx + 224 * 224 * 1] = ((color[1] / 255.f) - 0.456f) / 0.224f;
                    buf[idx + 224 * 224 * 0] = ((color[2] / 255.f) - 0.485f) / 0.229f;
                }
            }

            // std::cout << label << '\n'
            // cv::imshow("titke", out);
            // cv::waitKey(0);

            Ort::MemoryInfo meminfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            std::vector<std::int64_t> input_shape({1, 3, 224, 224});
            std::vector<std::int64_t> labels_shape({1});
            std::vector<Ort::Value> uinput;

            std::int32_t lab = 0;
            if (label == "dog")
                lab = 0;
            else if (label == "cat")
                lab = 1;
            else if (label == "cow")
                lab = 2;
            else
                lab = 3;

            uinput.emplace_back(
                Ort::Value::CreateTensor(
                    meminfo, buf, 3 * 224 * 224 * 4, input_shape.data(), input_shape.size(),
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

            uinput.emplace_back(
                Ort::Value::CreateTensor(
                    meminfo, &lab, 4, labels_shape.data(), labels_shape.size(),
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32));

            loss += *session.TrainStep(uinput).front().GetTensorMutableData<float>();

            session.OptimizerStep();
            session.LazyResetGrad();
        }

        printf("Epoch(%4d) loss: %f\n", i + 1, loss / 80);
    }

    free(buf);

    return 0;
}