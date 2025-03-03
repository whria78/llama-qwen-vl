#include "arg.h"
#include "base64.hpp"
#include "log.h"
#include "common.h"
#include "sampling.h"
#include "clip.h"
#include "llava.h"
#include "llama.h"
#include "ggml.h"

// WHRIA
#include <iostream>
#include <iomanip>  // 추가 필요
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <map>
#include <algorithm>
#include "TinyEXIF.h"
#include <onnxruntime_cxx_api.h>
#include <codecvt>
#include <locale>
#include <iostream>
#include <fstream>
#include <cmath>
#include <regex>

#include "nlohmann/json.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <string>

#include "TinyEXIF.h"
#include <iostream>
#include <fstream>
#include <string>


#include <iostream>
#include <cstdio>
#include <memory>
#include <array>
#include <string>
#include <cstdlib>
std::string GetVCRedistVersion() {
    std::string command = "reg query \"HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\VisualStudio\\14.0\\VC\\Runtimes\\x64\" /v Version";
    std::array<char, 128> buffer;
    std::string result;

    // Windows에서는 popen 대신 _popen, pclose 대신 _pclose 사용
    std::unique_ptr<FILE, decltype(&_pclose)> pipe(_popen(command.c_str(), "r"), _pclose);
    if (!pipe) return "Not Found";

    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }

    // 결과에서 버전 값만 추출
    std::istringstream iss(result);
    std::string line;
    while (std::getline(iss, line)) {
        if (line.find("Version") != std::string::npos) {
            std::istringstream lineStream(line);
            std::string temp, version;
            lineStream >> temp >> temp >> version;  // "Version" 이후의 값이 버전 번호
            return version;
        }
    }

    return "Not Found";
}

void CheckAndUpdateVCRedist() {
    std::string version = GetVCRedistVersion();
    std::string requiredVersion = "14.32.34438";

    if (version == "Not Found") {
        std::cout << "Visual C++ Redistributable not found.\n"
                  << "Please install it from:\n"
                  << "https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist" << std::endl;
    } else {
        std::cout << "Installed Visual C++ Redistributable version: " << version << std::endl;
    }
}

namespace fs = std::filesystem;

struct ImageInfo {
    std::string filePath;
    std::string dateTime;
};


std::string getExifDateTime(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file) {
        std::cerr  << "█ WHRIA: Cannot open file: " << filePath  << std::endl;
        return "";
    }

    // TinyEXIF를 사용하여 EXIF 데이터 읽기
    TinyEXIF::EXIFInfo exif(file);
    if (!exif.Fields) {
        std::cerr  << "█ WHRIA: No EXIF data found in: " << filePath  << std::endl;
        return "";
    }

    return exif.DateTimeOriginal;
}


static std::string image_to_base64(const std::string &fname) {
    cv::Mat img = cv::imread(fname);
    if (img.empty()) {
        std::cerr  << "█ WHRIA: Failed to load image: " << fname  <<std::endl;
        return "";
    }
    
    // 원본 이미지 크기 확인
    int original_width = img.cols;
    int original_height = img.rows;
    if (original_width == 0 || original_height == 0) {
        std::cerr  << "█ WHRIA: Invalid image dimensions: " << fname  <<std::endl;
        return "";
    }
    
    // 긴 변을 1000으로 리사이즈 (비율 유지)
    int new_width, new_height;
    if (original_width > original_height) {
        new_width = 1000;
        new_height = static_cast<int>((1000.0 / original_width) * original_height);
    } else {
        new_height = 1000;
        new_width = static_cast<int>((1000.0 / original_height) * original_width);
    }
    cv::resize(img, img, cv::Size(new_width, new_height));
    
    // 이미지 인코딩
    std::vector<uchar> buf;
    if (!cv::imencode(".jpg", img, buf)) {
        std::cerr  << "█ WHRIA: Failed to encode image: " << fname  <<std::endl;
        return "";
    }
    
    // Base64 변환
    auto required_bytes = base64::required_encode_size(buf.size());
    if (required_bytes == 0) {
        std::cerr  << "█ WHRIA: Base64 encoding size calculation failed: " << fname  <<std::endl;
        return "";
    }

    std::string base64_encoded;
    base64::encode(buf.begin(), buf.end(), std::back_inserter(base64_encoded));

    return base64_encoded;
}

using namespace cv;
using namespace std;


wstring string_to_wstring(const string& str) {
    wstring_convert<codecvt_utf8_utf16<wchar_t>> converter;
    return converter.from_bytes(str);
}

bool file_exists(const string& filename) {
    ifstream file(filename);
    return file.good();
}

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <filesystem>

std::vector<std::string> model_paths = {
        "efficientnet_lite0.onnx",
        "mobilenetv3_large_100.onnx",
    };

vector<float> softmax(const vector<float>& input) {
    vector<float> output(input.size());
    float max_val = *max_element(input.begin(), input.end());
    float sum = 0.0f;
    
    for (float val : input) {
        sum += exp(val - max_val);
    }
    
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = exp(input[i] - max_val) / sum;
    }
    
    return output;
}

Mat preprocess_image(const string& image_path) {
    Mat img = imread(image_path);
    if (img.empty()) {
        cerr << "Error: Could not read the image!" << endl;
        return {};
    }

    int h = img.rows;
    int w = img.cols;
    int crop_size = min(h, w);
    Rect roi((w - crop_size) / 2, (h - crop_size) / 2, crop_size, crop_size);
    img = img(roi);

    // 224x224 리사이징
    resize(img, img, Size(224, 224));

    // BGR → RGB 변환
    cvtColor(img, img, COLOR_BGR2RGB);

    // 히스토그램 평활화 적용 (채널별)
    vector<Mat> channels(3);
    split(img, channels);
    for (auto& ch : channels) {
        equalizeHist(ch, ch);
    }
    merge(channels, img);

    // float32 변환 (0~1 범위)
    img.convertTo(img, CV_32F, 1.0 / 255.0);

    // ImageNet 정규화 (Python 코드와 동일)
    vector<float> mean = {0.485, 0.456, 0.406};
    vector<float> std = {0.229, 0.224, 0.225};
    
    split(img, channels);
    for (int i = 0; i < 3; i++) {
        channels[i] = (channels[i] - mean[i]) / std[i];
    }
    merge(channels, img);

    // OpenCV dnn을 위한 blob 변환
    Mat blob;
    dnn::blobFromImage(img, blob);
    return blob;
}


vector<float> run_onnx_model(const string& model_path, const Mat& img) {
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime");
        Ort::SessionOptions session_options;
        std::wstring w_model_path = std::wstring(model_path.begin(), model_path.end());
        Ort::Session session(env, w_model_path.c_str(), session_options);
        
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name = session.GetInputNameAllocated(0, allocator);
        auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

        for (auto& dim : input_shape) {
            if (dim < 0) dim = 1;
        }

        vector<float> input_tensor_values(img.begin<float>(), img.end<float>());
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

        vector<const char*> output_names;
        for (size_t i = 0; i < session.GetOutputCount(); ++i) {
            output_names.push_back(session.GetOutputNameAllocated(i, allocator).get());
        }

        const char* input_name_cstr = input_name.get();
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr}, 
            &input_name_cstr,  // 수정된 부분
            &input_tensor, 
            1, 
            output_names.data(), 
            output_names.size()
        );
        
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        return softmax(vector<float>(output_data, output_data + output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount()));
    } catch (const Ort::Exception& e) {
        cerr << "Error: ONNX Runtime failed - " << e.what() << endl;
        return {};
    }
}

vector<float> run_multiple_onnx_models(const string& image_path, const vector<string>& model_paths) {
    Mat img = preprocess_image(image_path);
    if (img.empty()) return {};
    
    vector<vector<float>> outputs;
    for (const auto& model_path : model_paths) {
        outputs.push_back(run_onnx_model(model_path, img));
    }
    
    if (outputs.empty() || outputs[0].empty()) return {};
    
    vector<float> averaged_output(outputs[0].size(), 0.0f);
    for (const auto& output : outputs) {
        for (size_t i = 0; i < output.size(); ++i) {
            averaged_output[i] += output[i];
        }
    }
    
    for (float& val : averaged_output) {
        val /= outputs.size();
    }
    
    return averaged_output;
}


using json = nlohmann::json;
json jsonArray;

std::string extract_json(const std::string& response) {
    // 개행 문자를 포함하여 { ... } 패턴을 찾음
    std::regex json_regex(R"(\{[\s\S]*\})");  
    std::smatch match;

    if (std::regex_search(response, match, json_regex)) {
        return match.str();  // 첫 번째 { ... } 반환
    }

    return "";  // JSON이 없으면 빈 문자열 반환
}


#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef NDEBUG
#include "ggml-alloc.h"
#include "ggml-backend.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>


static bool qwen2vl_eval_image_embed(llama_context * ctx_llama, const struct llava_image_embed * image_embed,
                                     int n_batch, int * n_past, int * st_pos_id, struct clip_image_size * image_size) {
    int n_embd  = llama_model_n_embd(llama_get_model(ctx_llama));
    const int patch_size = 14 * 2;
    const int ph = image_size->height / patch_size + (image_size->height % patch_size > 0);
    const int pw = image_size->width / patch_size + (image_size->width % patch_size > 0);
    auto img_tokens = image_embed->n_image_pos;
    // llama_pos mrope_pos[img_tokens * 4];
    std::vector<llama_pos> mrope_pos;
    mrope_pos.resize(img_tokens * 4);

    for (int y = 0; y < ph; y++)
    {
        for (int x = 0; x < pw; x++)
        {
            int i = y * pw + x;
            mrope_pos[i] = *st_pos_id;
            mrope_pos[i + img_tokens] = *st_pos_id + y;
            mrope_pos[i + img_tokens * 2] = *st_pos_id + x;
            mrope_pos[i + img_tokens * 3] = 0;
        }
    }
    *st_pos_id += std::max(pw, ph);

    int processed = 0;
    std::vector<llama_pos> batch_mrope_pos;
    batch_mrope_pos.resize(img_tokens * 4);

    for (int i = 0; i < img_tokens; i += n_batch) {
        int n_eval = img_tokens - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }

        // llama_pos batch_mrope_pos[n_eval * 4];
        std::fill(batch_mrope_pos.begin(), batch_mrope_pos.end(), 0);
        memcpy(batch_mrope_pos.data(), &mrope_pos[processed], n_eval * sizeof(llama_pos));
        memcpy(&batch_mrope_pos[n_eval * 1], &mrope_pos[img_tokens * 1 + processed], n_eval * sizeof(llama_pos));
        memcpy(&batch_mrope_pos[n_eval * 2], &mrope_pos[img_tokens * 2 + processed], n_eval * sizeof(llama_pos));
        memcpy(&batch_mrope_pos[n_eval * 3], &mrope_pos[img_tokens * 3 + processed], n_eval * sizeof(llama_pos));

        llama_batch batch = {
            int32_t(n_eval),                // n_tokens
            nullptr,                        // token
            (image_embed->embed+i*n_embd),  // embed
            batch_mrope_pos.data(),         // pos
            nullptr,  // n_seq_id
            nullptr,  // seq_id
            nullptr,  // logits
        };

        if (llama_decode(ctx_llama, batch)) {
            LOG_ERR("%s : failed to eval\n", __func__);
            return false;
        }
        *n_past += n_eval;
        processed += n_eval;
    }
    return true;
}


static bool eval_tokens(struct llama_context * ctx_llama, std::vector<llama_token> tokens, int n_batch, int * n_past, int * st_pos_id) {
    int N = (int) tokens.size();
    std::vector<llama_pos> pos;
    for (int i = 0; i < N; i += n_batch) {
        int n_eval = (int) tokens.size() - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        auto batch = llama_batch_get_one(&tokens[i], n_eval);
        // TODO: add mrope pos ids somewhere else
        pos.resize(batch.n_tokens * 4);
        std::fill(pos.begin(), pos.end(), 0);
        for (int j = 0; j < batch.n_tokens * 3; j ++) {
            pos[j] = *st_pos_id + (j % batch.n_tokens);
        }
        batch.pos = pos.data();

        if (llama_decode(ctx_llama, batch)) {
            LOG_ERR("%s : failed to eval. token %d/%d (batch size %d, n_past %d)\n", __func__, i, N, n_batch, *n_past);
            return false;
        }
        *n_past += n_eval;
        *st_pos_id += n_eval;
    }
    return true;
}

static bool eval_id(struct llama_context * ctx_llama, int id, int * n_past, int * st_pos_id) {
    std::vector<llama_token> tokens;
    tokens.push_back(id);
    return eval_tokens(ctx_llama, tokens, 1, n_past, st_pos_id);
}

static bool eval_string(struct llama_context * ctx_llama, const char* str, int n_batch, int * n_past, int * st_pos_id, bool add_bos){
    std::string              str2     = str;
    std::vector<llama_token> embd_inp = common_tokenize(ctx_llama, str2, add_bos, true);
    eval_tokens(ctx_llama, embd_inp, n_batch, n_past, st_pos_id);
    return true;
}

static const char * sample(struct common_sampler * smpl,
                           struct llama_context * ctx_llama,
                           int * n_past, int * st_pos_id) {
    const llama_token id = common_sampler_sample(smpl, ctx_llama, -1);
    common_sampler_accept(smpl, id, true);

    const llama_model * model = llama_get_model(ctx_llama);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    static std::string ret;
    if (llama_vocab_is_eog(vocab, id)) {
        ret = "</s>";
    } else {
        ret = common_token_to_piece(ctx_llama, id);
    }
    eval_id(ctx_llama, id, n_past, st_pos_id);
    return ret.c_str();
}

static const char* IMG_BASE64_TAG_BEGIN = "<img src=\"data:image/jpeg;base64,";
static const char* IMG_BASE64_TAG_END = "\">";

static void find_image_tag_in_prompt(const std::string& prompt, size_t& begin_out, size_t& end_out) {
    begin_out = prompt.find(IMG_BASE64_TAG_BEGIN);
    end_out = prompt.find(IMG_BASE64_TAG_END, (begin_out == std::string::npos) ? 0UL : begin_out);
}

static bool prompt_contains_image(const std::string& prompt) {
    size_t begin, end;
    find_image_tag_in_prompt(prompt, begin, end);
    return (begin != std::string::npos);
}

// replaces the base64 image tag in the prompt with `replacement`
static llava_image_embed * llava_image_embed_make_with_prompt_base64(struct clip_ctx * ctx_clip, int n_threads, const std::string& prompt) {
    size_t img_base64_str_start, img_base64_str_end;
    find_image_tag_in_prompt(prompt, img_base64_str_start, img_base64_str_end);
    if (img_base64_str_start == std::string::npos || img_base64_str_end == std::string::npos) {
        LOG_ERR("%s: invalid base64 image tag. must be %s<base64 byte string>%s\n", __func__, IMG_BASE64_TAG_BEGIN, IMG_BASE64_TAG_END);
        return NULL;
    }

    auto base64_bytes_start = img_base64_str_start + strlen(IMG_BASE64_TAG_BEGIN);
    auto base64_bytes_count = img_base64_str_end - base64_bytes_start;
    auto base64_str = prompt.substr(base64_bytes_start, base64_bytes_count );

    auto required_bytes = base64::required_encode_size(base64_str.size());
    auto img_bytes = std::vector<unsigned char>(required_bytes);
    base64::decode(base64_str.begin(), base64_str.end(), img_bytes.begin());

    auto embed = llava_image_embed_make_with_bytes(ctx_clip, n_threads, img_bytes.data(), img_bytes.size());
    if (!embed) {
        LOG_ERR("%s: could not load image from base64 string.\n", __func__);
        return NULL;
    }

    return embed;
}

static std::string remove_image_from_prompt(const std::string& prompt, const char * replacement = "") {
    size_t begin, end;
    find_image_tag_in_prompt(prompt, begin, end);
    if (begin == std::string::npos || end == std::string::npos) {
        return prompt;
    }
    auto pre = prompt.substr(0, begin);
    auto post = prompt.substr(end + strlen(IMG_BASE64_TAG_END));
    return pre + replacement + post;
}

struct llava_context {
    struct clip_ctx * ctx_clip = NULL;
    struct llama_context * ctx_llama = NULL;
    struct llama_model * model = NULL;
};

static void print_usage(int, char ** argv) {
    LOG("\n example usage:\n");
    LOG("\n     %s -m <llava-v1.5-7b/ggml-model-q5_k.gguf> --mmproj <llava-v1.5-7b/mmproj-model-f16.gguf> --image <path/to/an/image.jpg> --image <path/to/another/image.jpg> [--temp 0.1] [-p \"describe the image in detail.\"]\n", argv[0]);
    LOG("\n note: a lower temperature value like 0.1 is recommended for better quality.\n");
}

static struct llava_image_embed * load_image(llava_context * ctx_llava, common_params * params, const std::string & fname) {

    // load and preprocess the image
    llava_image_embed * embed = NULL;
    auto prompt = params->prompt;
    if (prompt_contains_image(prompt)) {
        if (!params->image.empty()) {
            LOG_INF("using base64 encoded image instead of command line image path\n");
        }
        embed = llava_image_embed_make_with_prompt_base64(ctx_llava->ctx_clip, params->cpuparams.n_threads, prompt);
        if (!embed) {
            LOG_ERR("%s: can't load image from prompt\n", __func__);
            return NULL;
        }
        params->prompt = remove_image_from_prompt(prompt);
    } else {

        /*
        embed = llava_image_embed_make_with_filename(ctx_llava->ctx_clip, params->cpuparams.n_threads, fname.c_str());
        if (!embed) {
            fprintf(stderr, "%s: is %s really an image file?\n", __func__, fname.c_str());
            return NULL;
        }
        */

        // fname을 base64로 변환 & resize
        std::cerr  << "█ " << fname  <<std::endl;
        std::cerr  << "█ WHRIA: Start processing - " <<std::endl;
        std::string base64_image = image_to_base64(fname);
        std::cerr  << "█ WHRIA: Finished Base64 encoding"  <<std::endl;
        std::cerr  << "█ WHRIA: Running the VL model. At least 64GB of RAM is required. If a GPU of 1050 Ti or higher is available, the process takes approximately 2 to 5 minutes. If only a CPU is used, it takes about 5 to 10 minutes."  <<std::endl;

        if (base64_image.empty()) {
            fprintf(stderr, "%s: failed to convert image to base64: %s\n", __func__, fname.c_str());
            return NULL;
        }
        
        auto required_bytes = base64::required_encode_size(base64_image.size());
        auto img_bytes = std::vector<unsigned char>(required_bytes);
        base64::decode(base64_image.begin(), base64_image.end(), img_bytes.begin());
   
        embed = llava_image_embed_make_with_bytes(ctx_llava->ctx_clip, params->cpuparams.n_threads, img_bytes.data(), img_bytes.size());


        if (!embed) {
            fprintf(stderr, "%s: is %s really an image file?\n", __func__, fname.c_str());
            return NULL;
        }

    }

    return embed;
}

static std::string process_prompt(struct llava_context * ctx_llava, struct llava_image_embed * image_embed, common_params * params, const std::string & prompt) {

    cout << "█ PROMPT: " << prompt << endl;

    int n_past = 0;
    int cur_pos_id = 0;

    const int max_tgt_len = params->n_predict < 0 ? 256 : params->n_predict;

    std::string system_prompt, user_prompt;
    size_t image_pos = prompt.find("<|vision_start|>");
    if (image_pos != std::string::npos) {
        // new templating mode: Provide the full prompt including system message and use <image> as a placeholder for the image
        system_prompt = prompt.substr(0, image_pos);
        user_prompt = prompt.substr(image_pos + std::string("<|vision_pad|>").length());
        LOG_INF("system_prompt: %s\n", system_prompt.c_str());
        if (params->verbose_prompt) {
            auto tmp = common_tokenize(ctx_llava->ctx_llama, system_prompt, true, true);
            for (int i = 0; i < (int) tmp.size(); i++) {
                LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx_llava->ctx_llama, tmp[i]).c_str());
            }
        }
        LOG_INF("user_prompt: %s\n", user_prompt.c_str());
        if (params->verbose_prompt) {
            auto tmp = common_tokenize(ctx_llava->ctx_llama, user_prompt, true, true);
            for (int i = 0; i < (int) tmp.size(); i++) {
                LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx_llava->ctx_llama, tmp[i]).c_str());
            }
        }
    } else {
        // llava-1.5 native mode
        system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>";
        user_prompt = "<|vision_end|>" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
        if (params->verbose_prompt) {
            auto tmp = common_tokenize(ctx_llava->ctx_llama, user_prompt, true, true);
            for (int i = 0; i < (int) tmp.size(); i++) {
                LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx_llava->ctx_llama, tmp[i]).c_str());
            }
        }
    }

    eval_string(ctx_llava->ctx_llama, system_prompt.c_str(), params->n_batch, &n_past, &cur_pos_id, true);
    if (image_embed != nullptr) {
        auto image_size = clip_get_load_image_size(ctx_llava->ctx_clip);
        qwen2vl_eval_image_embed(ctx_llava->ctx_llama, image_embed, params->n_batch, &n_past, &cur_pos_id, image_size);
    }
    eval_string(ctx_llava->ctx_llama, user_prompt.c_str(), params->n_batch, &n_past, &cur_pos_id, false);

    // generate the response

    LOG("\n");

    struct common_sampler * smpl = common_sampler_init(ctx_llava->model, params->sampling);
    if (!smpl) {
        LOG_ERR("%s: failed to initialize sampling subsystem\n", __func__);
        exit(1);
    }

    std::string response = "";
    for (int i = 0; i < max_tgt_len; i++) {
        const char * tmp = sample(smpl, ctx_llava->ctx_llama, &n_past, &cur_pos_id);
        response += tmp;
        if (strcmp(tmp, "</s>") == 0) break;
        if (strstr(tmp, "###")) break; // Yi-VL behavior
        LOG("%s", tmp);
        if (strstr(response.c_str(), "<|im_end|>")) break; // Yi-34B llava-1.6 - for some reason those decode not as the correct token (tokenizer works)
        if (strstr(response.c_str(), "<|im_start|>")) break; // Yi-34B llava-1.6
        if (strstr(response.c_str(), "USER:")) break; // mistral llava-1.6

        fflush(stdout);
    }

    common_sampler_free(smpl);
    LOG("\n");


    return response;
}

static struct llama_model * llava_init(common_params * params) {
    llama_backend_init();
    llama_numa_init(params->numa);

    llama_model_params model_params = common_model_params_to_llama(*params);

    llama_model * model = llama_model_load_from_file(params->model.c_str(), model_params);
    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n" , __func__);
        return NULL;
    }
    return model;
}

static struct llava_context * llava_init_context(common_params * params, llama_model * model) {
    const char * clip_path = params->mmproj.c_str();

    auto prompt = params->prompt;
    if (prompt.empty()) {
        prompt = "describe the image in detail.";
    }

    auto ctx_clip = clip_model_load(clip_path, /*verbosity=*/ 1);

    llama_context_params ctx_params = common_context_params_to_llama(*params);
    ctx_params.n_ctx           = params->n_ctx < 2048 ? 2048 : params->n_ctx; // we need a longer context size to process image embeddings

    llama_context * ctx_llama = llama_init_from_model(model, ctx_params);

    if (ctx_llama == NULL) {
        LOG_ERR("%s: failed to create the llama_context\n" , __func__);
        return NULL;
    }

    auto * ctx_llava = (struct llava_context *)malloc(sizeof(llava_context));

    ctx_llava->ctx_llama = ctx_llama;
    ctx_llava->ctx_clip = ctx_clip;
    ctx_llava->model = model;
    return ctx_llava;
}

static void llava_free(struct llava_context * ctx_llava) {
    if (ctx_llava->ctx_clip) {
        clip_free(ctx_llava->ctx_clip);
        ctx_llava->ctx_clip = NULL;
    }

    llama_free(ctx_llava->ctx_llama);
    llama_model_free(ctx_llava->model);
    llama_backend_free();
}

#ifndef NDEBUG

static void debug_test_mrope_2d() {
    // 1. Initialize backend
    ggml_backend_t backend = NULL;
    std::string backend_name = "";
#ifdef GGML_USE_CUDA
    fprintf(stderr, "%s: using CUDA backend\n", __func__);
    backend = ggml_backend_cuda_init(0); // init device 0
    backend_name = "cuda";
    if (!backend) {
        fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
    }
#endif
    // if there aren't GPU Backends fallback to CPU backend
    if (!backend) {
        backend = ggml_backend_cpu_init();
        backend_name = "cpu";
    }

    // Calculate the size needed to allocate
    size_t ctx_size = 0;
    ctx_size += 2 * ggml_tensor_overhead(); // tensors
    // no need to allocate anything else!

    // 2. Allocate `ggml_context` to store tensor data
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_backend_alloc_ctx_tensors()
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * inp_raw = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 12, 30);
    ggml_set_name(inp_raw, "inp_raw");
    ggml_set_input(inp_raw);

    struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 30 * 4);
    ggml_set_name(pos, "pos");
    ggml_set_input(pos);

    std::vector<float> dummy_q;
    dummy_q.resize(128 * 12 * 30);
    std::fill(dummy_q.begin(), dummy_q.end(), 0.1);
    // memcpy(inp_raw->data, dummy_q.data(), 128 * 12 * 30 * ggml_element_size(inp_raw));

    std::vector<int> pos_id;
    pos_id.resize(30 * 4);
    for (int i = 0; i < 30; i ++) {
        pos_id[i] = i;
        pos_id[i + 30] = i + 10;
        pos_id[i + 60] = i + 20;
        pos_id[i + 90] = i + 30;
    }
    int sections[4] = {32, 32, 0, 0};

    // 4. Allocate a `ggml_backend_buffer` to store all tensors
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    // 5. Copy tensor data from main memory (RAM) to backend buffer
    ggml_backend_tensor_set(inp_raw, dummy_q.data(), 0, ggml_nbytes(inp_raw));
    ggml_backend_tensor_set(pos, pos_id.data(), 0, ggml_nbytes(pos));

    // 6. Create a `ggml_cgraph` for mul_mat operation
    struct ggml_cgraph * gf = NULL;
    struct ggml_context * ctx_cgraph = NULL;

    // create a temporally context to build the graph
    struct ggml_init_params params0 = {
        /*.mem_size   =*/ ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
    };
    ctx_cgraph = ggml_init(params0);
    gf = ggml_new_graph(ctx_cgraph);

    struct ggml_tensor * result0 = ggml_rope_multi(
        ctx_cgraph, inp_raw, pos, nullptr,
        128/2, sections, LLAMA_ROPE_TYPE_VISION, 32768, 1000000, 1,
        0, 1, 32, 1);

    // Add "result" tensor and all of its dependencies to the cgraph
    ggml_build_forward_expand(gf, result0);

    // 7. Create a `ggml_gallocr` for cgraph computation
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    // 9. Run the computation
    int n_threads = 1; // Optional: number of threads to perform some operations with multi-threading
    if (ggml_backend_is_cpu(backend)) {
        ggml_backend_cpu_set_n_threads(backend, n_threads);
    }
    ggml_backend_graph_compute(backend, gf);

    // 10. Retrieve results (output tensors)
    // in this example, output tensor is always the last tensor in the graph
    struct ggml_tensor * result = result0;
    // struct ggml_tensor * result = gf->nodes[gf->n_nodes - 1];
    float * result_data = (float *)malloc(ggml_nbytes(result));
    // because the tensor data is stored in device buffer, we need to copy it back to RAM
    ggml_backend_tensor_get(result, result_data, 0, ggml_nbytes(result));
    const std::string bin_file = "mrope_2d_" + backend_name +".bin";
    std::ofstream outFile(bin_file, std::ios::binary);

    if (outFile.is_open()) {
        outFile.write(reinterpret_cast<const char*>(result_data), ggml_nbytes(result));
        outFile.close();
        std::cout << "Data successfully written to " + bin_file << std::endl;
    } else {
        std::cerr << "Error opening file!" << std::endl;
    }

    free(result_data);
    // 11. Free memory and exit
    ggml_free(ctx_cgraph);
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
}

static void debug_dump_img_embed(struct llava_context * ctx_llava) {
    int n_embd  = llama_model_n_embd(llama_get_model(ctx_llava->ctx_llama));
    int ne = n_embd * 4;
    float vals[56 * 56 * 3];
    // float embd[ne];
    std::vector<float> embd;
    embd.resize(ne);

    for (int i = 0; i < 56*56; i++)
    {
        for (int c = 0; c < 3; c++)
            vals[i * 3 + c] = (float)(i % (56 * 56)) / (56*56);
    }

    clip_encode_float_image(ctx_llava->ctx_clip, 16, vals, 56, 56, embd.data());

    std::ofstream outFile("img_embed.bin", std::ios::binary);
    if (outFile.is_open()) {
        outFile.write(reinterpret_cast<const char*>(embd.data()), ne * sizeof(float));

        outFile.close();
        std::cout << "Data successfully written to mrope.bin" << std::endl;
    } else {
        std::cerr << "Error opening file!" << std::endl;
    }
}

#endif



int main(int argc, char ** argv) {

    CheckAndUpdateVCRedist();

    ggml_time_init();

    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_LLAVA, print_usage)) {
        return 1;
    }

    common_init();

    if (params.mmproj.empty()) {
        print_usage(argc, argv);
        return 1;
    }

    auto * model = llava_init(&params);
    if (model == NULL) {
        fprintf(stderr, "%s: error: failed to init llava model\n", __func__);
        return 1;
    }

    if (prompt_contains_image(params.prompt)) {
        auto * ctx_llava = llava_init_context(&params, model);

        auto * image_embed = load_image(ctx_llava, &params, "");

        // process the prompt
        std::string response = process_prompt(ctx_llava, image_embed, &params, params.prompt);

        llama_perf_context_print(ctx_llava->ctx_llama);
        llava_image_embed_free(image_embed);
        ctx_llava->model = NULL;
        llava_free(ctx_llava);
#ifndef NDEBUG
    } else if (params.image[0].empty()) {
        auto ctx_llava = llava_init_context(&params, model);

        debug_test_mrope_2d();
        debug_dump_img_embed(ctx_llava);

        llama_perf_context_print(ctx_llava->ctx_llama);
        ctx_llava->model = NULL;
        llava_free(ctx_llava);
#endif
    } else {
        for (auto & image_folder : params.image) {






            if (fs::is_regular_file(image_folder)) {
                std::cout << image_folder << " is a file." << std::endl;




                    std::string image=image_folder;

                    std::string date_=getExifDateTime(image);
                    std::cerr  << "█ WHRIA: DateTimeOriginal: " << date_  <<std::endl;            


                    vector<float> result = run_multiple_onnx_models(image,model_paths);

                    cout << "█ WHRIA: Averaged Model Output: ";
                    cout << std::fixed << std::setprecision(4);  // 소수점 4자리 고정

                    for (const auto& val : result) {
                        cout << val << " | ";
                    }
                    cout << endl;
                    
                    // 결과 출력 (2개 클래스 값 출력)
                    cout << "█ WHRIA: Clinical Photo: " << result[0] << endl;
                    cout << "█ WHRIA: Index Photo: " << result[1] << endl;


                    auto * ctx_llava = llava_init_context(&params, model);

                    auto * image_embed = load_image(ctx_llava, &params, image);
                    if (!image_embed) {
                        LOG_ERR("%s: failed to load image %s. Terminating\n\n", __func__, image.c_str());
                        return 1;
                    }

                    // process the prompt
                    std::string response = process_prompt(ctx_llava, image_embed, &params, params.prompt);
                    response=extract_json(response);
                    std::cout  << "█ WHRIA: Extracted JSON:\n" << response  <<std::endl;




                    llama_perf_context_print(ctx_llava->ctx_llama);
                    llava_image_embed_free(image_embed);
                    ctx_llava->model = NULL;
                    llava_free(ctx_llava);
                    






            }
            else        {

                std::vector<ImageInfo> images;

                for (const auto& entry : fs::recursive_directory_iterator(image_folder)) {
                    if (entry.is_regular_file()) {
                        std::string extension = entry.path().extension().string();
                        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
                        if (extension == ".jpg" || extension == ".jpeg" || extension == ".png") {
                            std::string filePath = entry.path().string();
                            std::replace(filePath.begin(), filePath.end(), '\\', '/');
                            std::string dateTime = getExifDateTime(filePath);
                            if (!dateTime.empty()) {
                                images.push_back({filePath, dateTime});
                            }
                            else
                            {
                                cout << "█ WHRIA: No EXIF infomation - " << filePath << endl;
                            }
                        }
                    }
                }

                // 날짜순 정렬
                std::sort(images.begin(), images.end(), [](const ImageInfo& a, const ImageInfo& b) {
                    return a.dateTime < b.dateTime;
                });

                int fileCount = 0;

                for (const auto& image_info : images) {
                    std::cerr << image_info.dateTime << " " << image_info.filePath << std::endl;
                    std::string image=image_info.filePath;

                    fileCount++;

                    vector<float> result = run_multiple_onnx_models(image,model_paths);
                    
                    cout << "█ WHRIA: Averaged Model Output: ";
                    cout << std::fixed << std::setprecision(4);  // 소수점 4자리 고정

                    for (const auto& val : result) {
                        cout << val << " | ";
                    }
                    cout << endl;

                    if (result[1]<params.onnx_threshold)
                        cout  << "█ WHRIA: CNNs predict that it is a Clinical Photo. "  <<endl;
                    else
                    {
                        cout  << "█ WHRIA: CNNs predict that it is a Index Photo. "  << endl;


                        // 기존 JSON 파일을 읽고 배열로 변환
                        //json jsonArray;
                        std::string out_json_path=(std::filesystem::path(image_folder) / std::filesystem::path(std::filesystem::path(image_folder).filename().string()+".json")).string();
                        std::replace(out_json_path.begin(), out_json_path.end(), '\\', '/');

                        std::ifstream inFile(out_json_path);
                        if (inFile.is_open()) {
                            try {
                                inFile >> jsonArray;
                                if (jsonArray.is_object()) { // 기존 파일이 객체 `{}` 형태라면 배열로 변환
                                    json tmpArray = json::array();
                                    tmpArray.push_back(jsonArray);
                                    jsonArray = tmpArray;
                                } else if (!jsonArray.is_array()) { // 객체도 배열도 아닌 경우 예외 처리
                                    throw std::runtime_error("Invalid JSON format: Expected array or object");
                                }
                            } catch (const std::exception &e) {
                                std::cerr  << "█ WHRIA: JSON file read error: " << e.what() << ". Initializing empty array."  <<std::endl;
                                jsonArray = json::array(); // 오류 발생 시 빈 배열 생성
                            }
                            inFile.close();
                        } else {
                            jsonArray = json::array(); // 파일이 없으면 빈 배열 생성
                        }

                        bool skip_ = false;
                        for (auto &item : jsonArray) {
                            if (item["Filename"] == image) {
                                skip_ = true;
                                break;
                            }
                        }
                        if (skip_ || (result[1]<params.onnx_threshold)) continue; // done phots or clnicalphoto


                        std::string response;
                        std::string confirm;
                        std::string err;

                        try 
                        {
                        
                            auto * ctx_llava = llava_init_context(&params, model);

                            auto * image_embed = load_image(ctx_llava, &params, image);
                            if (!image_embed) {
                                LOG_ERR("%s: failed to load image %s. Terminating\n\n", __func__, image.c_str());
                                return 1;
                            }

                            confirm = process_prompt(ctx_llava, image_embed, &params, "Does it include patient's name and registration number? Response must be YES or NO.");
                            
                            // response를 소문자로 변환
                            std::transform(confirm.begin(), confirm.end(), confirm.begin(), ::tolower);
                            // "yes"가 포함되어 있으면 bConfirm을 true로 설정
                            if (confirm.find("yes") != std::string::npos) confirm="yes"; else confirm="";

                            std::cout << std::endl;

                            std::cout  << "█ WHRIA: Does it include patient's name and registration number? Response must be YES or NO. : " << confirm   << std::endl;

                            llama_perf_context_print(ctx_llava->ctx_llama);

                            llava_image_embed_free(image_embed);
                            ctx_llava->model = NULL;
                            llava_free(ctx_llava);
                            
                            ctx_llava = llava_init_context(&params, model);
                            image_embed = load_image(ctx_llava, &params, image);
                            
                            // process the prompt
                            response = process_prompt(ctx_llava, image_embed, &params, params.prompt);

                            llama_perf_context_print(ctx_llava->ctx_llama);
                            llava_image_embed_free(image_embed);
                            ctx_llava->model = NULL;
                            llava_free(ctx_llava);

                            response=extract_json(response);
                            std::cout  << "█ WHRIA: Extracted JSON:\n" << response  <<std::endl;

                        } catch (const std::exception &e) {
                                cerr << e.what() << endl;
                                err=e.what();
                        }

                        json jsonObj; // 개별 JSON 객체

                        try {
                            jsonObj = json::parse(response);
                            jsonObj["response"] = response;
                            jsonObj["err"] = err;
                            jsonObj["Filename"] = image;
                            jsonObj["Name"] = jsonObj.value("Name", "");
                            jsonObj["ID"] = jsonObj.value("ID", "");
                            jsonObj["Date"] = image_info.dateTime;
                            jsonObj["confirm"] = confirm;
                            
                        } catch (const json::parse_error &e) {
                            std::cerr  << "█ WHRIA: JSON Parsing Error: " << e.what()  <<std::endl;
                            jsonObj["err"] = e.what();
                            jsonObj["response"] = response;
                            jsonObj["Filename"] = image;
                            jsonObj["Date"] = image_info.dateTime;
                            jsonObj["confirm"] = confirm;
                        } catch (const std::exception &e) {
                            std::cerr  << "█ WHRIA: Unknown Error: " << e.what()  <<std::endl;
                            jsonObj["err"] = e.what();
                            jsonObj["response"] = response;
                            jsonObj["Filename"] = image;
                            jsonObj["Date"] = image_info.dateTime;
                            jsonObj["confirm"] = confirm;
                        }



                        // Filename 중복 여부 확인 후 업데이트 또는 추가
                        bool found = false;
                        for (auto &item : jsonArray) {
                            if (item["Filename"] == jsonObj["Filename"]) {
                                item = jsonObj; // 기존 항목 덮어쓰기
                                found = true;
                                break;
                            }
                        }

                        if (!found) {
                            jsonArray.push_back(jsonObj); // 새로운 항목 추가
                        }

                        // 파일에 덮어쓰기 (새로운 JSON 추가된 전체 배열 저장)
                        std::ofstream outFile(out_json_path);
                        if (outFile.is_open()) {
                            outFile << jsonArray.dump(4); // JSON 배열을 파일에 저장 (들여쓰기 4칸)
                            outFile.close();
                            std::cout  << "█ WHRIA: JSON appended to " << out_json_path  << std::endl;
                        } else {
                            std::cerr  << "█ WHRIA: Failed to open " << out_json_path << " for writing"  <<std::endl;
                        }






                    }
                }



            }



                
            
        }
        

        
    }
    llama_model_free(model);

if (params.organize_photo)
{
    for (auto & image_folder : params.image) {

        std::vector<ImageInfo> images;
        map<string, json> jsonMap;
        json lastJsonEntry;
        string lastJsonTime;

        int iDone=0;
        int iNotDone=0;

        // 1. 이미지 파일 검색 및 EXIF 데이터 추출
        for (const auto& entry : fs::recursive_directory_iterator(image_folder)) {
            if (entry.is_regular_file()) {
                string extension = entry.path().extension().string();
                transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
                if (extension == ".jpg" || extension == ".jpeg" || extension == ".png") {
                    string filePath = entry.path().string();
                    replace(filePath.begin(), filePath.end(), '\\', '/');
                    string dateTime = getExifDateTime(filePath);
                    if (!dateTime.empty()) {
                        images.push_back({filePath, dateTime});
                    } else {
                        cout << "█ WHRIA: No EXIF information - " << filePath << endl;

                        fs::path targetFolder = fs::current_path() / "RESULT_NOTDONE";
                        std::string newFilename = fs::path(filePath).filename().string();

                        fs::create_directories(targetFolder);
                        fs::path destinationPath = fs::path(targetFolder) / newFilename;
                        
                        try {
                            fs::copy_file(filePath, destinationPath, fs::copy_options::overwrite_existing);
                            cout << "█ WHRIA: File copied to " << destinationPath.generic_string() << endl;
                        } catch (const exception &e) {
                            cerr << "█ WHRIA: Failed to copy file: " << e.what() << endl;
                        }
                        
                    }
                }
            }
        }

        // 2. 날짜순 정렬
        sort(images.begin(), images.end(), [](const ImageInfo& a, const ImageInfo& b) {
            return a.dateTime < b.dateTime;
        });

        // 3. JSON 파일 읽기
        std::string out_json_path = (fs::path(image_folder) / fs::path(fs::path(image_folder).filename().string() + ".json")).string();
        std::replace(out_json_path.begin(), out_json_path.end(), '\\', '/');

        ifstream inFile(out_json_path);
        if (inFile.is_open()) {
            try {
                json jsonArray;
                inFile >> jsonArray;
                if (jsonArray.is_object()) {
                    json tmpArray = json::array();
                    tmpArray.push_back(jsonArray);
                    jsonArray = tmpArray;
                }
                for (const auto& item : jsonArray) {
                    if (item.contains("Filename") && item.contains("Date")) {
                        jsonMap[item["Filename"].get<string>()] = item;
                    }
                }
            } catch (const exception& e) {
                cerr << "█ WHRIA: JSON file read error: " << e.what() << endl;
            }
            inFile.close();
        }


        // 시간 gap 에 맞춰서 Json 정보 추가


        for (const auto& image : images) {

            
            string sourcePath = image.filePath;
            string dateTime = image.dateTime;

            if (jsonMap.find(sourcePath) != jsonMap.end()) {
                lastJsonEntry = jsonMap[sourcePath];
                lastJsonTime = lastJsonEntry["Date"].get<string>();

                string id = lastJsonEntry.value("ID", "");
                string name = lastJsonEntry.value("Name", "");
            }

            bool useLastJson = false;
            if (!lastJsonTime.empty() && !dateTime.empty()) {
                struct tm tm1 = {}, tm2 = {};
                istringstream ss1(lastJsonTime), ss2(dateTime);
                if ((ss1 >> get_time(&tm1, "%Y:%m:%d %H:%M:%S")) && (ss2 >> get_time(&tm2, "%Y:%m:%d %H:%M:%S"))) {
                    time_t t1 = mktime(&tm1);
                    time_t t2 = mktime(&tm2);
                    if (t1 != -1 && t2 != -1) {
                        long diff = abs(difftime(t2, t1));
                        string id = lastJsonEntry.value("ID", "");
                        string name = lastJsonEntry.value("Name", "");
                        useLastJson = (diff <= params.organize_photo_timegap && (!id.empty() || !name.empty()) && lastJsonEntry.value("confirm", "")=="yes");  // params.organize_photo_timegap = 1200
                    }
                }
            }

            if (useLastJson && image.filePath!=lastJsonEntry.value("Filename", "")) {
                json jsonObj;
                jsonObj["Filename"] = image.filePath;
                jsonObj["Name"] = lastJsonEntry.value("Name", "");
                jsonObj["ID"] = lastJsonEntry.value("ID", "");


                // Filename 중복 여부 확인 후 업데이트 또는 추가
                bool found = false;
                for (auto &item : jsonArray) {
                    if (item["Filename"] == jsonObj["Filename"]) {
                        item = jsonObj; // 기존 항목 덮어쓰기
                        found = true;
                        break;
                    }
                }

                if (!found) {
                    jsonArray.push_back(jsonObj); // 새로운 항목 추가
                }

                std::string out_json_path=(std::filesystem::path(image_folder) / std::filesystem::path(std::filesystem::path(image_folder).filename().string()+".json")).string();
                std::replace(out_json_path.begin(), out_json_path.end(), '\\', '/');
                

                // 파일에 덮어쓰기 (새로운 JSON 추가된 전체 배열 저장)
                std::ofstream outFile(out_json_path);
                if (outFile.is_open()) {
                    outFile << jsonArray.dump(4); // JSON 배열을 파일에 저장 (들여쓰기 4칸)
                    outFile.close();
                    std::cout  << "█ WHRIA: JSON [" << image.filePath << "] appended to " << out_json_path  <<std::endl;
                } else {
                    std::cerr  << "█ WHRIA: Failed to open " << out_json_path << " for writing"  <<std::endl;
                }



            }
        }

        int fileIndex=0;
        // 4. 파일 복사 및 이름 변경 처리
        for (const auto& image : images) {
            string sourcePath = image.filePath;
            string dateTime = image.dateTime;
            string targetFolder;
            string newFilename;

            if (jsonMap.find(sourcePath) != jsonMap.end()) {
                lastJsonEntry = jsonMap[sourcePath];
                lastJsonTime = lastJsonEntry["Date"].get<string>();

                string id = lastJsonEntry.value("ID", "");
                string name = lastJsonEntry.value("Name", "");
                if (!id.empty() || !name.empty()) fileIndex=0;
            }

            bool useLastJson = false;
            if (!lastJsonTime.empty() && !dateTime.empty()) {
                struct tm tm1 = {}, tm2 = {};
                istringstream ss1(lastJsonTime), ss2(dateTime);
                if ((ss1 >> get_time(&tm1, "%Y:%m:%d %H:%M:%S")) && (ss2 >> get_time(&tm2, "%Y:%m:%d %H:%M:%S"))) {
                    time_t t1 = mktime(&tm1);
                    time_t t2 = mktime(&tm2);
                    if (t1 != -1 && t2 != -1) {
                        long diff = abs(difftime(t2, t1));
                        string id = lastJsonEntry.value("ID", "");
                        string name = lastJsonEntry.value("Name", "");
                        useLastJson = (diff <= params.organize_photo_timegap && (!id.empty() || !name.empty()) && lastJsonEntry.value("confirm", "")=="yes");  // params.organize_photo_timegap = 1200
                    }
                }
            }

            if (useLastJson) {
                string year = lastJsonTime.substr(0, 4);
                string month = lastJsonTime.substr(5, 2);
                targetFolder = (fs::current_path() / "RESULT" / year / (year + "-" + month)).string();
                string id = lastJsonEntry.value("ID", "");
                string name = lastJsonEntry.value("Name", "");
                string extension = fs::path(sourcePath).extension().string();

                fileIndex++;
                std::string number = std::to_string(fileIndex);
                number = std::string(6 - number.length(), '0') + number;
                
                newFilename = (!id.empty() || !name.empty()) ? id + (id.empty() || name.empty() ? "" : "_") + name + "_" + number + extension : fs::path(sourcePath).filename().string();
                
                iDone++;
                
            } else {
                string year = dateTime.substr(0, 4);
                string month = dateTime.substr(5, 2);
                targetFolder = (fs::current_path() / "RESULT_NOTDONE" / year / (year + "-" + month)).string();
                newFilename = fs::path(sourcePath).filename().string();
                
                iNotDone++;
               
            }

            fs::create_directories(targetFolder);
            fs::path destinationPath = fs::path(targetFolder) / newFilename;
            
            try {
                fs::copy_file(sourcePath, destinationPath, fs::copy_options::overwrite_existing);
                cout << "█ WHRIA: File copied to " << destinationPath.generic_string() << endl;
            } catch (const exception &e) {
                cerr << "█ WHRIA: Failed to copy file: " << e.what() << endl;
            }
        }

        cerr << "█ WHRIA: No. of Moved: " << iDone << endl;
        cerr << "█ WHRIA: No. of Failed: " << iNotDone << endl;
        
    }
}
if (params.copy_for_test) {
    fs::path targetFolder = fs::current_path() / "TEST";
    fs::create_directories(targetFolder);

    for (auto &image_folder : params.image) {
        std::vector<ImageInfo> images;
        std::map<std::string, json> jsonMap;

        for (const auto &entry : fs::recursive_directory_iterator(image_folder)) {
            if (entry.is_regular_file()) {
                std::string extension = entry.path().extension().string();
                std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
                if (extension == ".jpg" || extension == ".jpeg" || extension == ".png") {
                    std::string filePath = entry.path().string();
                    std::replace(filePath.begin(), filePath.end(), '\\', '/');
                    std::string dateTime = getExifDateTime(filePath);

                    if (!dateTime.empty()) {
                        images.push_back({filePath, dateTime});
                    } else {
                        std::cout << "█ WHRIA: No EXIF information - " << filePath << std::endl;
                    }
                }
            }
        }

        std::sort(images.begin(), images.end(), [](const ImageInfo &a, const ImageInfo &b) {
            return a.dateTime < b.dateTime;
        });

        std::string out_json_path = (fs::path(image_folder) / fs::path(fs::path(image_folder).filename().string() + ".json")).string();
        std::replace(out_json_path.begin(), out_json_path.end(), '\\', '/');
        std::ifstream inFile(out_json_path);
        if (inFile.is_open()) {
            try {
                json jsonArray;
                inFile >> jsonArray;
                if (jsonArray.is_object()) {
                    json tmpArray = json::array();
                    tmpArray.push_back(jsonArray);
                    jsonArray = tmpArray;
                }
                for (const auto &item : jsonArray) {
                    if (item.contains("Filename") && item.contains("Date")) {
                        jsonMap[item["Filename"].get<std::string>()] = item;
                    }
                }
            } catch (const std::exception &e) {
                std::cerr << "█ WHRIA: JSON file read error: " << e.what() << std::endl;
            }
            inFile.close();
        }

        int fileIndex = 0;
        json lastJsonEntry = {};
        std::string lastJsonTime;
        
        for (const auto &image : images) {
            std::string sourcePath = image.filePath;
            std::string dateTime = image.dateTime;
            std::string newFilename;

            std::string number = std::to_string(fileIndex);
            number = std::string(6 - number.length(), '0') + number;

            std::string filenameOnly = fs::path(sourcePath).filename().string();

            if (jsonMap.find(sourcePath) != jsonMap.end()) {
                json &entry = jsonMap[sourcePath];
                std::string id = entry.value("ID", "");
                std::string name = entry.value("Name", "");
                if (!id.empty() || !name.empty()) {
                    lastJsonEntry = entry;
                    lastJsonTime = entry.value("Date", "");
                }
            }

            if (!lastJsonTime.empty()) {
                struct tm tm1 = {}, tm2 = {};
                std::istringstream ss1(lastJsonTime), ss2(dateTime);
                if (!(ss1 >> std::get_time(&tm1, "%Y:%m:%d %H:%M:%S")) || !(ss2 >> std::get_time(&tm2, "%Y:%m:%d %H:%M:%S"))) {
                    std::cerr << "█ WHRIA: Error parsing time: " << lastJsonTime << " or " << dateTime << std::endl;
                } else {
                    time_t t1 = mktime(&tm1);
                    time_t t2 = mktime(&tm2);
                    if (t1 != -1 && t2 != -1 && std::abs(difftime(t2, t1)) <= params.organize_photo_timegap) {
                        std::string extension = fs::path(sourcePath).extension().string();
                        newFilename = lastJsonEntry.value("ID", "") + 
                                      (lastJsonEntry.value("ID", "").empty() || lastJsonEntry.value("Name", "").empty() ? "" : "_") + 
                                      lastJsonEntry.value("Name", "") + 
                                      extension;
                    } else {
                        newFilename = filenameOnly;
                    }
                }
            } else {
                newFilename = filenameOnly;
            }

            fs::path destinationPath = targetFolder / (number + "_" + newFilename);

            try {
                fs::copy_file(sourcePath, destinationPath, fs::copy_options::overwrite_existing);
                std::cout << "█ WHRIA: File copied to " << destinationPath.generic_string() << std::endl;
            } catch (const std::exception &e) {
                std::cerr << "█ WHRIA: Failed to copy file: " << e.what() << std::endl;
            }

            fileIndex++;
        }
    }
}





    return 0;
}
