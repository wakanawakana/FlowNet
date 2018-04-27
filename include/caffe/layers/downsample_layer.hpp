#ifndef CAFFE_DOWNSAMPLE_LAYER_HPP_
#define CAFFE_DOWNSAMPLE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"

namespace caffe {

	/**
	* @brief Phil's Downsample Layer
	* Takes a blob and downsamples width and height to given size
	*/
	template <typename Dtype>
	class DownsampleLayer : public Layer<Dtype> {
	public:
		explicit DownsampleLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "Downsample"; }
		virtual inline int MinBottomBlobs() const { return 1; }
		virtual inline int MaxBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }
		virtual inline bool AllowBackward() const { LOG(WARNING) << "DownsampleLayer does not do backward."; return false; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int count_;
		int num_;
		int channels_;
		int height_;
		int width_;

		int top_width_;
		int top_height_;
	};
}
#endif  // CAFFE_DOWNSAMPLE_LAYER_HPP_
