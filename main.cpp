/****************************************************************************
*   Generated by ACUITY 5.11.0
*   Match ovxlib 1.1.21
*
*   Neural Network application project entry file
****************************************************************************/
/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __linux__

#include <time.h>
#include <complex>
#include <unistd.h>

#elif defined(_WIN32)
#include <windows.h>
#include <vector>
#include <cmath>

#endif

#define RAND_INT(a, b) (rand() % ((b)-(a)+1))+ (a)

#include "vsi_nn_pub.h"
#include "vx_khr_cnn.h"

#include "vnn_global.h"
#include "vnn_pre_process.h"
#include "vnn_post_process.h"
#include "vnn_alike.h"
#include "timer.h"
#include "file.h"

#include <opencv2/opencv.hpp> // opencv

#include <sstream>
#include <dirent.h>
#include <SpRun.h>
#include <Tracker.h>

#define INPUT_META_NUM 1
#define RELEASE(x) {if(nullptr != (x)) free((x)); (x) = nullptr;}
static vnn_input_meta_t input_meta_tab[INPUT_META_NUM];
const std::string SUFFIX_IMAGE = "png bmp tiff tif jpg jpeg PNG JPG JPEG";
const int width = 640;
const int height = 384;
const int channel = 1;

/*-------------------------------------------
        Macros and Variables
-------------------------------------------*/

/*-------------------------------------------
                  Functions
-------------------------------------------*/
static void vnn_ReleaseNeuralNetwork
        (
                vsi_nn_graph_t *graph
        )
{
    vnn_ReleaseSuperpointV1(graph, TRUE);
    if (vnn_UseImagePreprocessNode())
    {
        vnn_ReleaseBufferImage();
    }
}

#define BILLION                                 1000000000

static uint64_t get_perf_count()
{
#if defined(__linux__) || defined(__ANDROID__) || defined(__QNX__) || defined(__CYGWIN__)
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);

    return (uint64_t) ((uint64_t) ts.tv_nsec + (uint64_t) ts.tv_sec * BILLION);
#elif defined(_WIN32) || defined(UNDER_CE)
    LARGE_INTEGER ln;

    QueryPerformanceCounter(&ln);

    return (uint64_t)ln.QuadPart;
#endif
}

static vsi_status vnn_VerifyGraph
        (
                vsi_nn_graph_t *graph
        )
{
    vsi_status status = VSI_FAILURE;
    uint64_t tmsStart, tmsEnd, msVal, usVal;

    /* Verify graph */
    printf("Verify...\n");
    tmsStart = get_perf_count();
    status = vsi_nn_VerifyGraph(graph);
    TEST_CHECK_STATUS(status, final);
    tmsEnd = get_perf_count();
    msVal = (tmsEnd - tmsStart) / 1000000;
    usVal = (tmsEnd - tmsStart) / 1000;
    printf("Verify Graph: %ldms or %ldus\n", msVal, usVal);

    final:
    return status;
}

static vsi_status vnn_ProcessGraph
        (
                vsi_nn_graph_t *graph
        )
{
    Timer timer;
    vsi_status status = VSI_FAILURE;
    int32_t i, loop;
    char *loop_s;
    float msVal, usVal;

    status = VSI_FAILURE;
    loop = 1; /* default loop time is 1 */
    loop_s = getenv("VNN_LOOP_TIME");
    if (loop_s)
    {
        loop = atoi(loop_s);
    }

    /* Run graph */
    for (i = 0; i < loop; i++)
    {
        status = vsi_nn_RunGraph(graph);
        if (status != VSI_SUCCESS)
        {
            printf("Run graph the %d time fail\n", i);
        }
        TEST_CHECK_STATUS(status, final);
    }


    timer.Timing("forward.", true);
    final:
    return status;
}

static void _load_input_meta()
{
    uint32_t i;
    for (i = 0; i < INPUT_META_NUM; i++)
    {
        memset(&input_meta_tab[i].image.preprocess, VNN_PREPRO_NONE,
                sizeof(int32_t) * VNN_PREPRO_NUM);
    }
    /* lid: input_256 */
    input_meta_tab[0].image.preprocess[0] = VNN_PREPRO_REORDER;
    input_meta_tab[0].image.preprocess[1] = VNN_PREPRO_MEAN;
    input_meta_tab[0].image.preprocess[2] = VNN_PREPRO_SCALE;
    input_meta_tab[0].image.reorder[0] = 0;
    input_meta_tab[0].image.reorder[1] = 1;
    input_meta_tab[0].image.reorder[2] = 2;
    input_meta_tab[0].image.mean[0] = 0.0;
    input_meta_tab[0].image.mean[1] = 0.0;
    input_meta_tab[0].image.mean[2] = 0.0;
    input_meta_tab[0].image.scale = 0.003921569;
}

static float *_imageData_to_float32(uint8_t *bmpData, vsi_nn_tensor_t *tensor)
{
    float *fdata;
    uint32_t sz, i;

    fdata = nullptr;
    sz = vsi_nn_GetElementNum(tensor);
    fdata = (float *) malloc(sz * sizeof(float));
    TEST_CHECK_PTR(fdata, final);

    for (i = 0; i < sz; i++)
    {
        fdata[i] = (float) bmpData[i];
    }

    final:
    return fdata;
}

static void _data_transform(float *fdata, vnn_input_meta_t *meta, vsi_nn_tensor_t *tensor)
{
    uint32_t s0, s1, s2;
    uint32_t i, j, offset, sz, order;
    float *data;
    uint32_t *reorder;

    data = nullptr;
    reorder = meta->image.reorder;
    s0 = tensor->attr.size[0];
    s1 = tensor->attr.size[1];
    s2 = tensor->attr.size[2];
    sz = vsi_nn_GetElementNum(tensor);
    data = (float *) malloc(sz * sizeof(float));
    TEST_CHECK_PTR(data, final);
    memset(data, 0, sizeof(float) * sz);

    for (i = 0; i < s2; i++)
    {
        if (s2 > 1 && reorder[i] <= s2)
        {
            order = reorder[i];
        }
        else
        {
            order = i;
        }

        offset = s0 * s1 * i;
        for (j = 0; j < s0 * s1; j++)
        {
            data[j + offset] = fdata[j * s2 + order];
        }
    }


    memcpy(fdata, data, sz * sizeof(float));
    final:
    if (data)free(data);
}

static void _data_mean(float *fdata, vnn_input_meta_t *meta, vsi_nn_tensor_t *tensor)
{
    uint32_t s0, s1, s2;
    uint32_t i, j, offset;
    float val, mean;

    s0 = tensor->attr.size[0];
    s1 = tensor->attr.size[1];
    s2 = tensor->attr.size[2];

    for (i = 0; i < s2; i++)
    {
        offset = s0 * s1 * i;
        mean = meta->image.mean[i];
        for (j = 0; j < s0 * s1; j++)
        {
            val = fdata[offset + j] - mean;
            fdata[offset + j] = val;
        }
    }

}

static void _data_scale(float *fdata, vnn_input_meta_t *meta, vsi_nn_tensor_t *tensor)
{
    uint32_t i, sz;
    float val, scale;

    sz = vsi_nn_GetElementNum(tensor);
    scale = meta->image.scale;
    if (0 != scale)
    {
        for (i = 0; i < sz; i++)
        {
            val = fdata[i] * scale;
            fdata[i] = val;
        }
    }
}

static uint8_t *_float32_to_dtype(float *fdata, vsi_nn_tensor_t *tensor)
{
    vsi_status status;
    uint8_t *data;
    uint32_t sz, i, stride;

    sz = vsi_nn_GetElementNum(tensor);
    stride = vsi_nn_TypeGetBytes(tensor->attr.dtype.vx_type);
    data = (uint8_t *) malloc(stride * sz * sizeof(uint8_t));
    TEST_CHECK_PTR(data, final);
    memset(data, 0, stride * sz * sizeof(uint8_t));

    for (i = 0; i < sz; i++)
    {
        status = vsi_nn_Float32ToDtype(fdata[i], &data[stride * i], &tensor->attr.dtype);
        if (status != VSI_SUCCESS)
        {
            if (data)free(data);
            return nullptr;
        }
    }

    final:
    return data;
}

static uint8_t *PreProcess(uint8_t *bmpData)
{
    using TYPE = uint8_t;
    TYPE *src = (TYPE *) bmpData;
    size_t size = width * height;
    int channel = 1;
    TYPE *ptr = (TYPE *) malloc(size * sizeof(TYPE));

    int offset, i, j;

//    #pragma omp parallel for
    for (i = 0; i < channel; i++)
    {
        offset = size * (channel - 1 - i);  // prapare input data
        for (j = 0; j < size; j++)
        {
            int tmpdata = (src[j * channel + i] >> 1);
            ptr[j + offset] = (TYPE) (tmpdata);
        }
    }

//    ptr = _float32_to_dtype(fdata, tensor);
    return ptr;
}


static uint8_t *PreProcess(vsi_nn_tensor_t *tensor, vnn_input_meta_t *meta, uint8_t *bmpData)
{
    uint8_t *data = nullptr;
    float *fdata = nullptr;
    int32_t use_image_process = vnn_UseImagePreprocessNode();

    TEST_CHECK_PTR(bmpData, final);
    if (use_image_process) return bmpData;
//    PrintMatrixUchar(bmpData, 400);
    fdata = _imageData_to_float32(bmpData, tensor);
//    PrintMatrix(fdata, 400);
    TEST_CHECK_PTR(fdata, final);

    for (uint32_t i = 0; i < _cnt_of_array(meta->image.preprocess); i++)
    {
        switch (meta->image.preprocess[i])
        {
            case VNN_PREPRO_NONE:
                break;
            case VNN_PREPRO_REORDER:
                _data_transform(fdata, meta, tensor);
                break;
            case VNN_PREPRO_MEAN:
                _data_mean(fdata, meta, tensor);
                break;
            case VNN_PREPRO_SCALE:
                _data_scale(fdata, meta, tensor);
                break;
            default:
                break;
        }
    }

//    PrintMatrix(fdata, 640);
    data = _float32_to_dtype(fdata, tensor);
    TEST_CHECK_PTR(data, final);
    final:
    RELEASE(fdata);

    return data;
}

static vsi_status UploadTensor(vsi_nn_graph_t *graph, uint8_t *data)
{
    const int inputID = 0;
    vsi_status status = VSI_FAILURE;
    vsi_nn_tensor_t *tensor = nullptr;
    char dumpInput[128];

    tensor = vsi_nn_GetTensor(graph, graph->input.tensors[inputID]);
    data = PreProcess(tensor, &(input_meta_tab[0]), data);
//    PrintMatrixUchar(data, 640);
    status = vsi_nn_CopyDataToTensor(graph, tensor, data);
    TEST_CHECK_STATUS(status, final);

    status = VSI_SUCCESS;
    final:
    return status;
}

vsi_status vnn_PreProcessNeuralNetworkSuperpoint(vsi_nn_graph_t *graph, cv::Mat dst)
{
    cv::Mat dst_gray;
    dst_gray = dst;


    _load_input_meta();
    vsi_status status = UploadTensor(graph, dst_gray.data);

    TEST_CHECK_STATUS(status, final);

    status = VSI_SUCCESS;
    final:
    return status;
}

static vsi_nn_graph_t *vnn_CreateNeuralNetwork(const char *data_file_name)
{
    vsi_nn_graph_t *graph = NULL;
    uint64_t tmsStart, tmsEnd, msVal, usVal;

    tmsStart = get_perf_count();
    graph = vnn_CreateAlileN(data_file_name, NULL, vnn_GetPrePorcessMap()
                                   , vnn_GetPrePorcessMapCount(), vnn_GetPostPorcessMap()
                                   , vnn_GetPostPorcessMapCount());
    TEST_CHECK_PTR(graph, final);
    tmsEnd = get_perf_count();
    msVal = (tmsEnd - tmsStart) / 1000000;
    usVal = (tmsEnd - tmsStart) / 1000;
    printf("Create Neural Network: %ldms or %ldus\n", msVal, usVal);

    final:
    return graph;
}

/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
template<class T>
uint64 Multiple(const T *data, const uint size)
{
    uint64 mul = 1;

    for (int i = 0; i < size; i++)
    {
        mul *= data[i];
    }

    return mul;
}

vsi_status GetOutput(vsi_nn_graph_t *graph, vsi_nn_tensor_t *tensor, Array &array, bool dequant = true)
{
    vsi_status status = VSI_FAILURE;
    array.dims = std::vector<uint>(std::begin(tensor->attr.size),
            std::begin(tensor->attr.size) + tensor->attr.dim_num);
    uint64 size = Multiple(tensor->attr.size, tensor->attr.dim_num);
    uint stride = vsi_nn_TypeGetBytes(tensor->attr.dtype.vx_type);
    uint8_t *tensor_data = (uint8_t *) vsi_nn_ConvertTensorToData(graph, tensor);
    array.data = (float *) malloc(sizeof(float) * size);

    if (dequant)
    {
        for (uint i = 0; i < size; i++)
        {
            status = vsi_nn_DtypeToFloat32(&tensor_data[stride * i], &array.data[i]
                    , &tensor->attr.dtype);
        }
    }
    else
    {
        uint8_t *tensor_int =  (uint8_t *) malloc(sizeof(uint8_t) * size);
        memcpy(tensor_int,  tensor_data, sizeof(uint8_t) * size);
    }

    if (tensor_data) vsi_nn_Free(tensor_data);

    return status;

}

void Array::Release()
{
    RELEASE(data);
}

Array::~Array()
{
    Release();
}

void norm(Array desc)
{
    cv::Mat image;
    int pixnums = desc.dims[0] * desc.dims[1];

    for (int i = 0; i < desc.dims[0]; i++)
    {
        for (int j = 0; j < desc.dims[1]; j++)
        {
            PrintMatrix(desc.data, 4000);
//            std::complex<float> data{desc.data};

        }
    }
}

void Reshape(Array &input, TensorType &output)
{
    int featureSize = input.dims[0] * input.dims[1];

    output.resize(input.dims[2], std::vector<float>(featureSize, 0));
    for (int i = 0; i < input.dims[2]; ++i)
    {
        int id = featureSize * i;
        memcpy(output.at(i).data(), input.data + id, featureSize * sizeof(float));
    }
}

void ReshapeSemi(Array &input, TensorType &output)
{
    int featureSize = input.dims[0];

    output.resize(input.dims[1], std::vector<float>(featureSize, 0));
    for (int i = 0; i < input.dims[1]; ++i)
    {
        int id = featureSize * i;
        memcpy(output.at(i).data(), input.data + id, featureSize * sizeof(float));
    }
}

TensorType InverseTransform(const TensorType &output, const int width)
{
    TensorType result = output;

    if (output.empty()) return result;
    if (output[0].empty()) return result;

    const size_t CHANNEL = output.size();
    const size_t featureSize = output[0].size();

    for (int i = 0; i < output.size(); ++i)
    {
        for (int j = 0; j < featureSize; ++j)
        {
            size_t id = i * featureSize + j;
            size_t channel = id % CHANNEL;
            size_t col = id / CHANNEL % width;
            size_t row = id / (CHANNEL * width);
            result[channel][row * width + col] = output[i][j];
        }
    }


    return result;
}

void Reshape(Array &input, float ***output)
{
    output = new float **[input.dims[3]];
    output[0] = new float *[input.dims[2]];

    int featureSize = input.dims[0] * input.dims[1];
    for (size_t channel = 0; channel < input.dims[2]; ++channel)
    {
        float *channelData = new float[featureSize];
        memcpy(channelData, input.data + channel * featureSize, featureSize * sizeof(float));
        output[0][channel] = channelData;
    }
}

void NormDiv(const Array &input, Array &output, const int p = 2, const int dim = 1)
{
    if (dim >= input.dims.size() or dim < 0)
    {
        std::cout << "input dim for Norm beyond 0~input,dims.size() " << std::endl;
    }
    int size = 1;
    int normSize = 1;
    for (int i = 0; i < input.dims.size(); ++i)
    {
        size *= input.dims[i];

        if (i == dim) continue;
        normSize *= input.dims[i];
    }

    if (size == 0)
    {
        std::cout << "error dim for Norm in Relocation, dim contain: " << size << std::endl;
        return;
    }

    output.dims = input.dims;
    output.data = new float[size];
    memcpy(output.data, input.data, size * sizeof(float));

    float *norm = new float[normSize]();

    if (dim == 1)
    {
        for (int i = 0; i < output.dims[0]; ++i)
        {
            for (int k = 0; k < output.dims[2]; ++k)
            {
                for (int j = 0; j < output.dims[1]; ++j)
                {
                    norm[i * output.dims[2] + k] +=
                            output.data[i * output.dims[1] * output.dims[2] + j * output.dims[2] +
                                        k]
                            * output.data[i * output.dims[1] * output.dims[2] + j * output.dims[2] +
                                          k];
                }
            }
        }
    }
    else if (dim == 0 or dim == 2)
    {
        for (int i = 0; i < normSize; ++i)
        {
            for (int j = 0; j < output.dims[dim]; ++j)
            {
                if (dim == 0)
                {
                    norm[i] += output.data[j * normSize + i] * output.data[j * normSize + i];
                }
                else if (dim == 2)
                {
                    norm[i] += output.data[i * output.dims[dim] + j] *
                               output.data[i * output.dims[dim] + j];
                }
            }
        }
    }
    else
    {
        std::cout << "input dim > 2, error!" << std::endl;
        return;
    }


    for (int i = 0; i < normSize; ++i)
    {
        norm[i] = sqrt(norm[i]);
    }

    if (dim == 1)
    {
        for (int i = 0; i < output.dims[0]; ++i)
        {
            for (int k = 0; k < output.dims[2]; ++k)
            {
                for (int j = 0; j < output.dims[1]; ++j)
                {
                    output.data[i * output.dims[1] * output.dims[2] + j * output.dims[2] + k] /=
                            norm[i * output.dims[2] + k];
                }
            }
        }
    }
    else if (dim == 0 or dim == 2)
    {
        for (int i = 0; i < normSize; ++i)
        {
            if (abs(norm[i]) - 0 < 1e-5)
            {
                std::cout << "zero is not for div" << std::endl;
                continue;
            }
            for (int j = 0; j < output.dims[dim]; ++j)
            {
                output.data[i * output.dims[dim] + j] /= norm[i];
            }
            std::cout << "norm[" << i << "] : " << norm[i];
        }
    }

    std::cout << std::endl;
    delete norm;
}

vsi_status PostProcess(vsi_nn_graph_t *graph, cv::Mat img
                       , Points &points, TensorType &describes)
{
    Timer timer, timerAll;
    uint64_t tmsStart = get_perf_count();
    vsi_status status = VSI_FAILURE;

    Array semi, coarse_desc;
    int outpixNum = 80 * 48;

    status = GetOutput(graph, vsi_nn_GetTensor(graph, graph->output.tensors[0]), semi);
    status |= GetOutput(graph, vsi_nn_GetTensor(graph, graph->output.tensors[1]), coarse_desc, true);

    timer.Timing("get output", true);
    if (status != VSI_SUCCESS) return status;

//    PrintMatrix(coarse_desc.data, 80);
    // torch.norm
//    norm(coarse_desc);

    long long width = 640;
    long long height = 384;
    TensorType semiResult, descResult;  // ( 1, 384*640) (128, 48*80)

    ReshapeSemi(semi, semiResult);
    Reshape(coarse_desc, descResult);

    timer.Timing("reshape output", true);
//    SpRun::Norm(descResult);
//    PrintMatrix(descResult[0].data(), 80);
    timer.Timing("normal.", true);

//    PrintMatrix(semiResult[0].data(), 80);

    SpRun *sp = new SpRun(coarse_desc.dims[2], outpixNum, height, width);
    sp->calc(semiResult, descResult, img, points, describes);
    timerAll.Timing("post process", true);

    return VSI_SUCCESS;
}

std::string GetSuffix(const char *fileName)
{
    const char SEPARATOR = '.';
    char buff[32] = {0};

    const char *ptr = strrchr(fileName, SEPARATOR);

    if (nullptr == ptr) return "";

    uint32_t pos = ptr - fileName;
    uint32_t n = strlen(fileName) - (pos + 1);
    strncpy(buff, fileName + (pos + 1), n);

    return buff;
}

bool IsIMAGE(const char *fileName)
{
    std::string suffix = GetSuffix(fileName);

    if (suffix.empty()) return false;
    return SUFFIX_IMAGE.find(suffix) != std::string::npos;
}


void Walk(const std::string &path, const std::string suffixList
          , std::vector<std::string> &fileList)
{
    DIR *dir;
    dir = opendir(path.c_str());
    struct dirent *ent;
    if (nullptr == dir)
    {
//        std::cout << "failed to open file " << path << std::endl;
        return;
    }

    while ((ent = readdir(dir)) != nullptr)
    {
        auto name = std::string(ent->d_name);

        // ignore "." ".."
        if (name.size() < 4) continue;

        std::string suffix = GetSuffix(name.c_str());

        if (!suffix.empty() && suffixList.find(suffix) != std::string::npos)
        {
            fileList.emplace_back(path + "/" + name);
        }
        else
        {
            Walk(path + "/" + name, suffixList, fileList);
        }

    }

    closedir(dir);
}

void RunModel(const cv::Mat &dst, vsi_nn_graph_t *graph
              , Points &points, TensorType &describes)
{
    vsi_status status = VSI_FAILURE;
    {
        /* Pre process the image data */
        status = vnn_PreProcessNeuralNetworkSuperpoint(graph, dst);
        TEST_CHECK_STATUS(status, final);

        //    /* Process graph */
        status = vnn_ProcessGraph(graph);
        TEST_CHECK_STATUS(status, final);

        status = PostProcess(graph, dst, points, describes);
        TEST_CHECK_STATUS(status, final);
    }

    return;

    final:
    std::cout << std::endl;
}

cv::Mat ConvertVectorMat(const TensorType &descResultLeft)
{
//    cv::Mat imageResult(0, descResultLeft[0].size(), cv::DataType<float>::type);
//    for (int i = 0; i < descResultLeft.size(); ++i)
//    {
//        cv::Mat Sample(1, descResultLeft[0].size(), cv::DataType<float>::type, descResultLeft[i].data());
//        imageResult.push_back(Sample);
//    }

    if (descResultLeft.size() > 0)
    {
        cv::Mat imageResult(descResultLeft.size(), descResultLeft[1].size(), CV_32F);
        for (int i = 0; i < descResultLeft.size(); ++i)
            imageResult.row(i) = cv::Mat(descResultLeft[i]).t();

        return imageResult;
    }
    else
    {
        std::cout << "describes.size() == 0" << std::endl;
        cv::Mat image;
        return image;
    }
}


void Gamma(cv::Mat &img, float gamma = 0.5)
{
    std::vector<int> table(256, 1);
    for (int i = 0; i < 256; ++i)
    {
        table[i] = int(pow(i / 255.0, gamma) * 255);
    }

    float gamma2 = gamma * 0.8;
    for (int i = 180; i < 256; ++i)
    {
        table[i] = int(pow(i / 255.0, gamma2) * 255);
    }

    cv::LUT(img, table, img);
    img.convertTo(img, CV_8U);
}

void Augment(cv::Mat &img)
{
    const int MIN_MEAN = 60;

    cv::Scalar mean, std;
    cv::meanStdDev(img, mean, std);
    std::cout << "mean: " << mean[0] << ", " << std[0] << std::endl;

    if (mean[0] < MIN_MEAN)
    {
        std::vector<cv::Mat> channel;
        cv::split(img, channel);
        cv::equalizeHist(channel[0], channel[0]);

        cv::merge(std::vector<cv::Mat>{channel[0], channel[0], channel[0]}, img);
    }
    Gamma(img, 0.8);
}

void GetFeature(cv::Mat& imageLeft, vsi_nn_graph_t *graph
                , Points &points, TensorType &describes)
{
    Timer timer;
    cv::Mat dstLeft;
//    Augment(imageLeft);
    cv::resize(imageLeft, dstLeft, cv::Size(width, height));
    RunModel(dstLeft, graph, points, describes);
    timer.Timing("alike.", true);
}


void ShowEveryPointMatch(cv::Mat imgLeft, cv::Mat imgRight
                         , const std::string extendName
                         , const Points &pointsLeft, const Points &pointsRight
                         , const std::vector<cv::DMatch> matches)
{
    cv::Mat imageLeftRight;
    SpRun::ShowImage("", pointsLeft, imgLeft);
    SpRun::ShowImage("", pointsRight, imgRight);
    cv::hconcat(imgLeft, imgRight, imageLeftRight);
    cv::Mat stereoImg;
    imageLeftRight.copyTo(stereoImg);

    char *p = getcwd(NULL, 0);
    std::string outputRoot = std::string(p) + "/" + "result" + "/" + extendName + "/";
    file_op::File::CreatDir(outputRoot);

    auto color = cv::Scalar(0, 0, 255);

    std::cout << "saving every line for match..." << std::endl;

    for (int i = 0; i < matches.size(); ++i)
    {
        cv::DMatch dMatch;
        dMatch = matches[i];
        const auto &left = pointsLeft[dMatch.queryIdx];
        const auto &right = pointsRight[dMatch.trainIdx];
        cv::Point leftCV = cv::Point(left.x, left.y);
        cv::Point rightCV = cv::Point(right.x + imgLeft.cols, right.y);

        cv::line(imageLeftRight, leftCV, rightCV, color, 1, cv::LINE_AA);
        int gap = 8;
        cv::line(imageLeftRight, cv::Point(leftCV.x, 0), cv::Point(leftCV.x, leftCV.y - gap), color
                 , 1, cv::LINE_AA);
        cv::line(imageLeftRight, cv::Point(leftCV.x, leftCV.y + gap), cv::Point(leftCV.x
                                                                                , stereoImg.rows)
                 , color, 1, cv::LINE_AA);
        cv::line(imageLeftRight, cv::Point(rightCV.x, 0), cv::Point(rightCV.x, rightCV.y - gap)
                 , color, 1, cv::LINE_AA);
        cv::line(imageLeftRight, cv::Point(rightCV.x, rightCV.y + gap), cv::Point(rightCV.x
                                                                                  , stereoImg.rows)
                 , color, 1, cv::LINE_AA);

        cv::imwrite(outputRoot + std::to_string(i) + ".jpg", imageLeftRight);
        stereoImg.copyTo(imageLeftRight);
    }
}

std::vector<cv::DMatch>
Match(const Points &semiResultLeft, const Points &semiResultRight, const TensorType &descResultLeft
      , const TensorType &descResultRight, std::vector<cv::DMatch> &matches)
{
    bool crossCheck = true;
    cv::BFMatcher matcher(cv::NORM_L2, crossCheck);
    cv::Mat descriptorLeft = ConvertVectorMat(descResultLeft);
    cv::Mat descriptorRight = ConvertVectorMat(descResultRight);

    matcher.match(descriptorLeft, descriptorRight, matches);
    std::cout << "point-L: " << semiResultLeft.size() << " point-R: "
              << semiResultRight.size() << " , matches: " << matches.size() << std::endl;
}

cv::Scalar GetColor(const float value)
{
    auto color = cv::Scalar(0, 255, 0);
    float ratio = MIN(1, value + 0.2);
    color[2] = 255 - ratio * 255;
    color[1] = ratio * 255;

    return color;
}

void ShowMatch(cv::Mat imgLeft, cv::Mat imgRight
               , const std::string extendName
               , const Points &pointsLeft, const Points &pointsRight
               , const std::vector<cv::DMatch> matches)
{
    cv::Mat imageLeftRight;
    int drawPoints = 0;
    const int W = imgLeft.cols, H = imgLeft.rows;

    SpRun::ShowImage("", pointsLeft, imgLeft);
    SpRun::ShowImage("", pointsRight, imgRight);
    cv::hconcat(imgLeft, imgRight, imageLeftRight);

    for (int i = 0; i < matches.size(); ++i)
    {
        cv::DMatch dMatch;
        dMatch = matches[i];
        const auto &left = pointsLeft[dMatch.queryIdx];
        const auto &right = pointsRight[dMatch.trainIdx];
        cv::Point leftCV = cv::Point(left.x, left.y);
        cv::Point rightCV = cv::Point(right.x + imgLeft.cols, right.y);

        drawPoints += 1;

        cv::line(imageLeftRight, leftCV, rightCV, cv::Scalar(RAND_INT(0, 255), RAND_INT(0, 255)
                                                             , RAND_INT(0, 255)), 1, cv::LINE_AA);
    }

    cv::putText(imageLeftRight, "match " + std::to_string(drawPoints) + "", cv::Point(50, H - 30)
                , cv::FONT_HERSHEY_COMPLEX
                , 1, cv::Scalar(0, 255, 0), 2);

    char *p = getcwd(NULL, 0);
    std::string outputFile = std::string(p) + "/" + "result";

    file_op::File::CreatDir(outputFile);

    outputFile += "/result_" + extendName + ".jpg";
    cv::imwrite(outputFile, imageLeftRight);

    std::cout << outputFile << std::endl;
}

void GetPairMatch(const std::string imageLeftFile, const std::string imageRightFile
                  , vsi_nn_graph_t *graph, bool saveLine)
{
    TensorType descResultLeft, descResultRight;
    Points semiResultLeft, semiResultRight;
    std::vector<cv::DMatch> matches;

    cv::Mat imageLeft = cv::imread(imageLeftFile);
    cv::Mat imageRight = cv::imread(imageRightFile);

    GetFeature(imageLeft, graph, semiResultLeft, descResultLeft);
//    SpRun::ShowImage(imageLeftFile, semiResultLeft, imageLeft);
    GetFeature(imageRight, graph, semiResultRight, descResultRight);
    Match(semiResultLeft, semiResultRight, descResultLeft, descResultRight, matches);

    int index = imageLeftFile.find_last_of('/');
    std::string path = imageLeftFile.substr(index + 1, -1);
    int index2 = path.find_last_of(".");
    std::string extendName = path.substr(0, index2);

    if (saveLine)
        ShowEveryPointMatch(imageLeft, imageRight, extendName, semiResultLeft, semiResultRight
                            , matches);
    else
        ShowMatch(imageLeft, imageRight, extendName, semiResultLeft, semiResultRight, matches);
}

void NormDivTest()
{
    cv::Mat image = cv::imread("/home/khadas/Parker/image.png");
    Array image_a, image_b;

//    std::cout << "==Numpy风格==\n" << cv::format(image, cv::Formatter::FMT_NUMPY) << std::endl;
    std::cout << image << std::endl;
    image_a.data = new float[75]();

    std::cout << "image.channels(): " << image.channels() << std::endl;
    for (int i = 0; i < image.rows; ++i)
    {
        for (int j = 0; j < image.cols; ++j)
        {
            for (int c = 0; c < image.channels(); ++c)
            {
                image_a.data[i * 5 * 3 + j * 3 + c] = image.at<cv::Vec3b>(i, j)[c];
            }

        }
    }
    std::cout << std::endl;
    for (int i = 0; i < 75; ++i)
    {
        std::cout << image_a.data[i] << ", ";
    }
    std::cout << std::endl;
    image_a.dims = {5, 5, 3};
    NormDiv(image_a, image_b);
    std::cout << "image_b: ";
    for (int i = 0; i < 75; ++i)
    {
        std::cout << image_b.data[i] << ", ";
    }
    std::cout << std::endl;
}

void FindReplace(std::string &str, const std::string orig, const std::string rep)
{
    size_t pos = str.find(orig);

    if (std::string::npos != pos)
    {
        str.replace(pos, orig.size(), rep);
        return;
    }
}

int main(int argc, char **argv)
{
//    NormDivTest();
    vsi_status status = VSI_FAILURE;
    vsi_nn_graph_t *graph;
    const char *data_name = NULL;
    bool saveLine = false;
    bool usePair = false;
    bool useSequence = false;

    if (argc < 3)
    {
        printf("Usage: %s [modle_file] [dataset]...\n", argv[0]);
        return -1;
    }

    if (argc > 3)
    {
        std::string flag = argv[3];
        if (flag == "stereo")
            usePair = true;
        else if (flag == "sequence")
            useSequence = true;
    }
    if (argc > 4) saveLine = true;

    data_name = (const char *) argv[1];

    std::vector<std::string> imageFiles({});
    char *path = (argv + 2)[0];

    if (IsIMAGE(path))
    {
        imageFiles.emplace_back(path);
    }
    else
    {

        Walk(path, SUFFIX_IMAGE, imageFiles);
        std::vector<std::string> imageFilesLeft({});

        for (const auto& file : imageFiles)
        {
            if (file.find("imgs.L") == std::string::npos
                and file.find("cam0") == std::string::npos)
                continue;

            imageFilesLeft.push_back(file);
        }

        imageFiles.swap(imageFilesLeft);

        std::sort(imageFiles.begin(), imageFiles.end());
    }
    graph = vnn_CreateNeuralNetwork(data_name);
    TEST_CHECK_PTR(graph, final);

    /* Verify graph */
    status = vnn_VerifyGraph(graph);
    TEST_CHECK_STATUS(status, final);

    for (size_t i = 0; i < imageFiles.size() - 1; ++i)
    {
        const auto& file = imageFiles[i];

        if (usePair)
        {
            std::string leftFile = file, rightFile = file;
            std::cout << std::endl;
            FindReplace(rightFile, "imgs.L", "imgs.R");
            FindReplace(rightFile, "_L", "_R");
            FindReplace(rightFile, "cam0", "cam1");

            GetPairMatch(leftFile, rightFile, graph, saveLine);
        }
        else if (useSequence)
        {
            std::string preFile = file;
            std::string nextFile = imageFiles[i + 1];
            cv::Mat preImg = cv::imread(preFile);

            float diff = 0;
            for (size_t j = i + 1; j < imageFiles.size(); ++j)
            {
                nextFile = imageFiles[j];

                cv::Mat nextImg = cv::imread(nextFile);
                cv::absdiff(preImg, nextImg, preImg);
                diff = cv::mean(preImg)[0];
                bool same = diff < 100;
                if (not same)
                {
                    i = j - 1;
                    break;
                }
            }

            std::cout << "diff: " << diff << std::endl;
            std::cout << "pair " << preFile << " " << nextFile << std::endl;
            GetPairMatch(preFile, nextFile, graph, saveLine);
        }
        else
        {
            TensorType describes;
            Points points;
            cv::Mat dst;
            cv::Mat image = cv::imread(file);

            GetFeature(image, graph, points, describes);
            SpRun::ShowImage(file, points, image);
            TEST_CHECK_STATUS(status, final);
        }
    }


    final:
    vnn_ReleaseNeuralNetwork(graph);
    fflush(stdout);
    fflush(stderr);
    return status;
}

