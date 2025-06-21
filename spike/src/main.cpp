#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <algorithm>
#include <numeric>
#include <string>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
}

#include <NvInfer.h>
#include <NvInferVersion.h>
#include <cuda_runtime.h>

// Simple struct to hold TensorRT engine data
struct TensorRTEngine {
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    
    void* gpu_input_buffer = nullptr;
    std::vector<void*> gpu_output_buffers;
    
    std::string input_tensor_name;
    std::vector<std::string> output_tensor_names;
    
    size_t input_buffer_size = 0;
    std::vector<size_t> output_buffer_sizes;
};

// Simple logger class
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "TensorRT: " << msg << std::endl;
        }
    }
};

static Logger g_logger;

// Static function to load TensorRT engine
static bool loadEngine(const std::string& engine_path, TensorRTEngine& engine_data) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Failed to open engine file: " << engine_path << std::endl;
        return false;
    }
    
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    
    std::vector<char> engine_buffer(size);
    file.read(engine_buffer.data(), size);
    file.close();
    
    engine_data.runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    if (!engine_data.runtime) {
        std::cerr << "Failed to create TensorRT runtime" << std::endl;
        return false;
    }
    
    engine_data.engine = std::unique_ptr<nvinfer1::ICudaEngine>(
        engine_data.runtime->deserializeCudaEngine(engine_buffer.data(), size)
    );
    if (!engine_data.engine) {
        std::cerr << "Failed to deserialize engine" << std::endl;
        return false;
    }
    
    engine_data.context = std::unique_ptr<nvinfer1::IExecutionContext>(
        engine_data.engine->createExecutionContext()
    );
    if (!engine_data.context) {
        std::cerr << "Failed to create execution context" << std::endl;
        return false;
    }
    
    // Get tensor information
    int num_bindings = engine_data.engine->getNbIOTensors();
    
    for (int i = 0; i < num_bindings; ++i) {
        const char* tensor_name = engine_data.engine->getIOTensorName(i);
        auto tensor_mode = engine_data.engine->getTensorIOMode(tensor_name);
        
        if (tensor_mode == nvinfer1::TensorIOMode::kINPUT) {
            engine_data.input_tensor_name = tensor_name;
        } else {
            engine_data.output_tensor_names.push_back(tensor_name);
        }
    }
    
    if (engine_data.input_tensor_name.empty() || engine_data.output_tensor_names.empty()) {
        std::cerr << "Failed to find input/output tensors" << std::endl;
        return false;
    }
    
    std::cout << "Engine loaded successfully!" << std::endl;
    std::cout << "Input: " << engine_data.input_tensor_name << std::endl;
    std::cout << "Outputs: ";
    for (const auto& name : engine_data.output_tensor_names) {
        std::cout << name << " ";
    }
    std::cout << std::endl;
    
    return true;
}

// Static function to cleanup GPU memory
static void cleanupEngine(TensorRTEngine& engine_data) {
    if (engine_data.gpu_input_buffer) {
        cudaFree(engine_data.gpu_input_buffer);
        engine_data.gpu_input_buffer = nullptr;
    }
    for (auto& buffer : engine_data.gpu_output_buffers) {
        if (buffer) {
            cudaFree(buffer);
            buffer = nullptr;
        }
    }
    engine_data.gpu_output_buffers.clear();
}


static std::vector<float> preprocessBatch(const std::vector<AVFrame*>& frames) {
    if (frames.empty()) {
        return {};
    }
    const int batch_size = frames.size();
    const int H = frames[0]->height;
    const int W = frames[0]->width;
    const int C = 3;

    std::vector<float> batch_data(batch_size * C * H * W);
    
    SwsContext* sws_ctx = sws_getContext(W, H, (AVPixelFormat)frames[0]->format, W, H, AV_PIX_FMT_RGB24, SWS_BILINEAR, nullptr, nullptr, nullptr);
    std::vector<uint8_t> rgb_buffer(W * H * C);
    uint8_t* dst[] = { rgb_buffer.data() };
    int dst_stride[] = { W * C };

    for (int i = 0; i < batch_size; ++i) {
        sws_scale(sws_ctx, (const uint8_t* const*)frames[i]->data, frames[i]->linesize, 0, H, dst, dst_stride);
        
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    size_t plane_offset = i * C * H * W + c * H * W;
                    batch_data[plane_offset + h * W + w] = rgb_buffer[(h * W + w) * C + c] / 255.0f;
                }
            }
        }
    }
    sws_freeContext(sws_ctx);
    return batch_data;
}

static std::vector<AVFrame*> postprocessBatch(
    const std::vector<std::vector<float>>& output_data,
    const TensorRTEngine& engine_data,
    int batch_size, int output_w, int output_h) {

    const auto& context = *engine_data.context;
    
    auto get_output_by_name = [&](const std::string& name) -> const std::vector<float>* {
        for (size_t i = 0; i < engine_data.output_tensor_names.size(); ++i) {
            if (engine_data.output_tensor_names[i] == name) {
                return &output_data[i];
            }
        }
        return nullptr;
    };

    const std::vector<float>* left_batch_ptr = get_output_by_name("left");
    const std::vector<float>* right_batch_ptr = get_output_by_name("right");
    
    if (!left_batch_ptr || !right_batch_ptr) {
        std::cerr << "Could not find 'left' and 'right' output tensors." << std::endl;
        return {};
    }
    const auto& left_batch = *left_batch_ptr;
    const auto& right_batch = *right_batch_ptr;

    auto left_dims = context.getTensorShape("left");
    int C = left_dims.d[1];
    int H = left_dims.d[2];
    int W = left_dims.d[3];
    int new_W = output_w / 2;

    std::vector<AVFrame*> processed_frames;
    processed_frames.reserve(batch_size);
    
    std::vector<uint8_t> combined_rgb(output_w * output_h * C);
    SwsContext* sws_ctx = sws_getContext(W, H, AV_PIX_FMT_RGB24, new_W, output_h, AV_PIX_FMT_RGB24, SWS_BILINEAR, nullptr, nullptr, nullptr);
    std::vector<uint8_t> side_buffer(W * H * C);

    for (int i = 0; i < batch_size; ++i) {
        // Process left
        for(int c=0; c < C; ++c) for(int h=0; h < H; ++h) for(int w=0; w < W; ++w) {
            side_buffer[(h * W + w) * C + c] = std::max(0, std::min(255, (int)(left_batch[i*C*H*W + c*H*W + h*W + w] * 255.0f)));
        }
        uint8_t* src[] = { side_buffer.data() };
        int src_stride[] = { W * C };
        uint8_t* dst_left[] = { combined_rgb.data() };
        int dst_stride_left[] = { output_w * C };
        sws_scale(sws_ctx, src, src_stride, 0, H, dst_left, dst_stride_left);

        // Process right
        for(int c=0; c < C; ++c) for(int h=0; h < H; ++h) for(int w=0; w < W; ++w) {
            side_buffer[(h * W + w) * C + c] = std::max(0, std::min(255, (int)(right_batch[i*C*H*W + c*H*W + h*W + w] * 255.0f)));
        }
        uint8_t* dst_right[] = { combined_rgb.data() + new_W * C };
        int dst_stride_right[] = { output_w * C };
        sws_scale(sws_ctx, src, src_stride, 0, H, dst_right, dst_stride_right);
        
        AVFrame* frame = av_frame_alloc();
        frame->format = AV_PIX_FMT_RGB24;
        frame->width = output_w;
        frame->height = output_h;
        av_frame_get_buffer(frame, 0);
        memcpy(frame->data[0], combined_rgb.data(), combined_rgb.size());
        processed_frames.push_back(frame);
    }
    sws_freeContext(sws_ctx);
    return processed_frames;
}

static std::vector<AVFrame*> runBatchInference(TensorRTEngine& engine_data, const std::vector<AVFrame*>& frames) {
    if (!engine_data.context || frames.empty()) {
        std::cerr << "Engine not initialized or empty frame batch" << std::endl;
        return {};
    }

    const int batch_size = frames.size();
    const int height = frames[0]->height;
    const int width = frames[0]->width;

    // Preprocess
    std::vector<float> input_data = preprocessBatch(frames);

    const char* input_name = engine_data.input_tensor_name.c_str();
    
    // Set input shape for dynamic models
    nvinfer1::Dims input_shape;
    input_shape.nbDims = 4;
    input_shape.d[0] = batch_size;
    input_shape.d[1] = 3;
    input_shape.d[2] = height;
    input_shape.d[3] = width;
    
    if (!engine_data.context->setInputShape(input_name, input_shape)) {
        std::cerr << "Failed to set input shape!" << std::endl;
        return {};
    }
    
    if (!engine_data.context->allInputDimensionsSpecified() || !engine_data.context->allInputShapesSpecified()) {
        std::cerr << "Not all input dimensions/shapes are specified!" << std::endl;
        return {};
    }

    // Allocate GPU buffers
    size_t actual_input_size = input_data.size() * sizeof(float);
    if (actual_input_size > engine_data.input_buffer_size) {
        if (engine_data.gpu_input_buffer) cudaFree(engine_data.gpu_input_buffer);
        cudaMalloc(&engine_data.gpu_input_buffer, actual_input_size);
        engine_data.input_buffer_size = actual_input_size;
    }

    engine_data.gpu_output_buffers.resize(engine_data.output_tensor_names.size());
    engine_data.output_buffer_sizes.resize(engine_data.output_tensor_names.size());
    std::vector<std::vector<float>> output_cpu_data(engine_data.output_tensor_names.size());

    for (size_t i = 0; i < engine_data.output_tensor_names.size(); ++i) {
        const char* output_name = engine_data.output_tensor_names[i].c_str();
        auto output_dims = engine_data.context->getTensorShape(output_name);
        
        size_t output_size = 1;
        for(int j=0; j<output_dims.nbDims; ++j) {
            output_size *= output_dims.d[j];
        }
        output_size *= sizeof(float);
        
        if (output_size > engine_data.output_buffer_sizes[i]) {
            if (engine_data.gpu_output_buffers[i]) cudaFree(engine_data.gpu_output_buffers[i]);
            cudaMalloc(&engine_data.gpu_output_buffers[i], output_size);
            engine_data.output_buffer_sizes[i] = output_size;
        }
        output_cpu_data[i].resize(output_size / sizeof(float));
    }
    
    cudaMemcpy(engine_data.gpu_input_buffer, input_data.data(), actual_input_size, cudaMemcpyHostToDevice);
    
    engine_data.context->setTensorAddress(input_name, engine_data.gpu_input_buffer);
    for (size_t i = 0; i < engine_data.output_tensor_names.size(); ++i) {
        engine_data.context->setTensorAddress(engine_data.output_tensor_names[i].c_str(), engine_data.gpu_output_buffers[i]);
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    if (!engine_data.context->enqueueV3(stream)) {
        std::cerr << "Inference failed" << std::endl;
        cudaStreamDestroy(stream);
        return {};
    }
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    
    for (size_t i = 0; i < engine_data.output_tensor_names.size(); ++i) {
        cudaMemcpy(output_cpu_data[i].data(), engine_data.gpu_output_buffers[i], engine_data.output_buffer_sizes[i], cudaMemcpyDeviceToHost);
    }
    
    return postprocessBatch(output_cpu_data, engine_data, batch_size, width, height);
}


static void encode_and_write(AVCodecContext *enc_ctx, AVFormatContext *ofmt_ctx, AVFrame *frame, SwsContext*& sws_ctx, AVPixelFormat target_pix_fmt) {
    AVFrame* sw_frame = frame;
    if (frame && frame->format != target_pix_fmt) {
        if (!sws_ctx) {
            sws_ctx = sws_getContext(frame->width, frame->height, (AVPixelFormat)frame->format,
                                     enc_ctx->width, enc_ctx->height, enc_ctx->pix_fmt,
                                     SWS_BILINEAR, NULL, NULL, NULL);
        }
        AVFrame* tmp_frame = av_frame_alloc();
        tmp_frame->format = enc_ctx->pix_fmt;
        tmp_frame->width = enc_ctx->width;
        tmp_frame->height = enc_ctx->height;
        av_frame_get_buffer(tmp_frame, 0);
        sws_scale(sws_ctx, (const uint8_t * const*)frame->data, frame->linesize, 0, frame->height, tmp_frame->data, tmp_frame->linesize);
        // Copy timestamp information
        tmp_frame->pts = frame->pts;
        tmp_frame->pkt_dts = frame->pkt_dts;
        tmp_frame->time_base = frame->time_base;
        tmp_frame->duration = frame->duration;
        sw_frame = tmp_frame;
    }
    
    int ret = avcodec_send_frame(enc_ctx, sw_frame);
    if (sw_frame != frame) av_frame_free(&sw_frame);

    if (ret < 0) {
        // For flushing, avcodec_send_frame will return an error once it's flushed, which is normal.
        // For regular frames, this is an error.
        if (frame) {
            std::cerr << "[encode_and_write] avcodec_send_frame failed, ret=" << ret << std::endl;
            return;
        }
    }

    while (true) {
        AVPacket *pkt = av_packet_alloc();
        ret = avcodec_receive_packet(enc_ctx, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            av_packet_free(&pkt);
            return;
        } else if (ret < 0) {
            std::cerr << "[encode_and_write] avcodec_receive_packet failed, ret=" << ret << std::endl;
            av_packet_free(&pkt);
            // This is a real error
            return;
        }
        av_packet_rescale_ts(pkt, enc_ctx->time_base, ofmt_ctx->streams[0]->time_base);
        pkt->stream_index = 0;
        int write_ret = av_interleaved_write_frame(ofmt_ctx, pkt);
        if (write_ret < 0) {
            std::cerr << "[encode_and_write] av_interleaved_write_frame failed, ret=" << write_ret << std::endl;
        }
        av_packet_free(&pkt);
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <engine_path> <input_video_path> <output_video_path> [batch_size]" << std::endl;
        return 1;
    }
    std::string engine_path = argv[1];
    std::string input_video_path = argv[2];
    std::string output_video_path = argv[3];
    int batch_size = (argc > 4) ? std::stoi(argv[4]) : 4;
    
    TensorRTEngine engine_data;
    if (!loadEngine(engine_path, engine_data)) {
        return 1;
    }

    // FFmpeg decoding setup
    AVFormatContext *ifmt_ctx = nullptr;
    if (avformat_open_input(&ifmt_ctx, input_video_path.c_str(), nullptr, nullptr) < 0) {
        std::cerr << "Could not open input file " << input_video_path << std::endl;
        return 1;
    }
    if (avformat_find_stream_info(ifmt_ctx, nullptr) < 0) {
        std::cerr << "Could not find stream info" << std::endl;
        return 1;
    }
    int video_stream_idx = av_find_best_stream(ifmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_stream_idx < 0) {
        std::cerr << "Could not find video stream" << std::endl;
        return 1;
    }
    AVCodecParameters* codecpar = ifmt_ctx->streams[video_stream_idx]->codecpar;
    const AVCodec* decoder = avcodec_find_decoder(codecpar->codec_id);
    if (!decoder) {
        std::cerr << "Could not find decoder for codec_id " << codecpar->codec_id << std::endl;
        return 1;
    }
    AVCodecContext* dec_ctx = avcodec_alloc_context3(decoder);
    if (!dec_ctx) {
        std::cerr << "Could not allocate decoder context" << std::endl;
        return 1;
    }
    if (avcodec_parameters_to_context(dec_ctx, codecpar) < 0) {
        std::cerr << "Could not copy codec parameters to decoder context" << std::endl;
        return 1;
    }
    if (avcodec_open2(dec_ctx, decoder, nullptr) < 0) {
        std::cerr << "Could not open decoder" << std::endl;
        return 1;
    }

    // FFmpeg encoding setup
    AVFormatContext *ofmt_ctx = nullptr;
    avformat_alloc_output_context2(&ofmt_ctx, nullptr, nullptr, output_video_path.c_str());
    const AVCodec *encoder = avcodec_find_encoder(AV_CODEC_ID_H264);
    AVStream *out_stream = avformat_new_stream(ofmt_ctx, encoder);
    AVCodecContext *enc_ctx = avcodec_alloc_context3(encoder);
    enc_ctx->height = dec_ctx->height;
    enc_ctx->width = dec_ctx->width;
    enc_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    enc_ctx->time_base = av_inv_q(ifmt_ctx->streams[video_stream_idx]->r_frame_rate);
    enc_ctx->bit_rate = 4000000;
    enc_ctx->gop_size = 12;
    enc_ctx->max_b_frames = 2;
    if (ofmt_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
        enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }
    av_opt_set(enc_ctx->priv_data, "preset", "fast", 0);

    if (avcodec_open2(enc_ctx, encoder, nullptr) < 0) {
        std::cerr << "Could not open encoder" << std::endl;
        return 1;
    }
    if (avcodec_parameters_from_context(out_stream->codecpar, enc_ctx) < 0) {
        std::cerr << "Could not copy encoder parameters to output stream" << std::endl;
        return 1;
    }
    if (avio_open(&ofmt_ctx->pb, output_video_path.c_str(), AVIO_FLAG_WRITE) < 0) {
        std::cerr << "Could not open output file" << std::endl;
        return 1;
    }
    if (avformat_write_header(ofmt_ctx, nullptr) < 0) {
        std::cerr << "Error occurred when opening output file" << std::endl;
        return 1;
    }

    AVPacket *pkt = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();
    std::vector<AVFrame*> frame_buffer;
    SwsContext *sws_ctx_out = nullptr;
    int frame_count = 0;
    
    while (av_read_frame(ifmt_ctx, pkt) >= 0) {
        if (pkt->stream_index == video_stream_idx) {
            if (avcodec_send_packet(dec_ctx, pkt) == 0) {
                while (avcodec_receive_frame(dec_ctx, frame) == 0) {
                    AVFrame* cloned_frame = av_frame_clone(frame);
                    frame_buffer.push_back(cloned_frame);

                    if (frame_buffer.size() >= batch_size) {
                        std::vector<AVFrame*> processed_frames = runBatchInference(engine_data, frame_buffer);
                        for (size_t i = 0; i < processed_frames.size(); i++) {
                            // Rescale timestamps from decoder to encoder timebase
                            AVRational in_tb = ifmt_ctx->streams[video_stream_idx]->time_base;
                            AVRational out_tb = enc_ctx->time_base;
                            processed_frames[i]->pts = av_rescale_q(frame_buffer[i]->pts, in_tb, out_tb);
                            if (frame_buffer[i]->pkt_dts != AV_NOPTS_VALUE) {
                                processed_frames[i]->pkt_dts = av_rescale_q(frame_buffer[i]->pkt_dts, in_tb, out_tb);
                            } else {
                                processed_frames[i]->pkt_dts = AV_NOPTS_VALUE;
                            }
                            if (frame_buffer[i]->duration > 0) {
                                processed_frames[i]->duration = av_rescale_q(frame_buffer[i]->duration, in_tb, out_tb);
                            } else {
                                processed_frames[i]->duration = 0; // Or calculate based on framerate if needed
                            }

                            encode_and_write(enc_ctx, ofmt_ctx, processed_frames[i], sws_ctx_out, enc_ctx->pix_fmt);
                            av_frame_free(&processed_frames[i]);
                        }
                        frame_count += frame_buffer.size();
                        std::cout << "Processed " << frame_count << " frames" << std::endl;
                        for(auto f : frame_buffer) av_frame_free(&f);
                        frame_buffer.clear();
                    }
                }
            }
        }
        av_packet_unref(pkt);
    }

    if (!frame_buffer.empty()) {
        std::vector<AVFrame*> processed_frames = runBatchInference(engine_data, frame_buffer);
        for (size_t i = 0; i < processed_frames.size(); i++) {
            // Rescale timestamps from decoder to encoder timebase
            AVRational in_tb = ifmt_ctx->streams[video_stream_idx]->time_base;
            AVRational out_tb = enc_ctx->time_base;
            processed_frames[i]->pts = av_rescale_q(frame_buffer[i]->pts, in_tb, out_tb);
            if (frame_buffer[i]->pkt_dts != AV_NOPTS_VALUE) {
                processed_frames[i]->pkt_dts = av_rescale_q(frame_buffer[i]->pkt_dts, in_tb, out_tb);
            } else {
                processed_frames[i]->pkt_dts = AV_NOPTS_VALUE;
            }
            if (frame_buffer[i]->duration > 0) {
                processed_frames[i]->duration = av_rescale_q(frame_buffer[i]->duration, in_tb, out_tb);
            } else {
                processed_frames[i]->duration = 0; // Or calculate based on framerate if needed
            }

            encode_and_write(enc_ctx, ofmt_ctx, processed_frames[i], sws_ctx_out, enc_ctx->pix_fmt);
            av_frame_free(&processed_frames[i]);
        }
        frame_count += frame_buffer.size();
        std::cout << "Processed " << frame_count << " frames (final batch)" << std::endl;
        for(auto f : frame_buffer) av_frame_free(&f);
        frame_buffer.clear();
    }
    
    encode_and_write(enc_ctx, ofmt_ctx, NULL, sws_ctx_out, enc_ctx->pix_fmt); // flush encoder
    if (av_write_trailer(ofmt_ctx) < 0) {
        std::cerr << "Error occurred when writing output trailer" << std::endl;
    }
    avio_closep(&ofmt_ctx->pb);
    avformat_free_context(ofmt_ctx);
    
    std::cout << "Finished processing. Output saved to " << output_video_path << std::endl;

    cleanupEngine(engine_data);
    avformat_close_input(&ifmt_ctx);
    avcodec_free_context(&dec_ctx);
    avcodec_free_context(&enc_ctx);
    av_packet_free(&pkt);
    av_frame_free(&frame);
    if (sws_ctx_out) sws_freeContext(sws_ctx_out);
    
    return 0;
}