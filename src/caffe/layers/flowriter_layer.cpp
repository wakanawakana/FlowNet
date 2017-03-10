// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <vector>
#include <cmath>

#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.h"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/net.hpp"
#include "caffe/solver.hpp"

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#include <iostream>
#include <fstream>
#include <omp.h>
#if defined (_MSC_VER)
#include <direct.h>
#include <windows.h>
#else
#include <sys/dir.h>
#endif

using std::max;

namespace caffe {

template <typename Dtype>
void FLOWriterLayer<Dtype>::writeFloFile(string filename, const float* data, int xSize, int ySize)
{
    FILE *stream = fopen(filename.c_str(), "wb");

    // write the header
    fprintf(stream,"PIEH");
    fwrite(&xSize,sizeof(int),1,stream);
    fwrite(&ySize,sizeof(int),1,stream);

    // write the data
    for (int y = 0; y < ySize; y++)
        for (int x = 0; x < xSize; x++) {
            float u = data[y*xSize+x];
            float v = data[y*xSize+x+ySize*xSize];
            fwrite(&u,sizeof(float),1,stream);
            fwrite(&v,sizeof(float),1,stream);
        }
    fclose(stream);
}
  
template <typename Dtype>
void FLOWriterLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{


}

template <typename Dtype>
void FLOWriterLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
    CHECK_EQ(bottom.size(), 1) << "FLOWRITER layer takes one input";

    const int channels = bottom[0]->channels();

    CHECK_EQ(channels, 2) << "FLOWRITER layer input must have two channels";

#if defined (_MSC_VER)
	HANDLE hFind;
	WIN32_FIND_DATA FindFileData;
	hFind = FindFirstFile((LPCWSTR)this->layer_param_.writer_param().folder().c_str(), &FindFileData);
	if (hFind != INVALID_HANDLE_VALUE){
		FindClose(hFind);
	} else {
		int retval = _mkdir(this->layer_param_.writer_param().folder().c_str());
	}
#else
	DIR* dir = opendir(this->layer_param_.writer_param().folder().c_str());
    if (dir)
        closedir(dir);
    else if (ENOENT == errno) {
		std::string cmd("mkdir -p " + this->layer_param_.writer_param().folder());
        int retval = std::system(cmd.c_str());
        (void)retval;
    }
#endif
}

template <typename Dtype>
void FLOWriterLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int height = bottom[0]->height();
    const int width = bottom[0]->width();

    Net<Dtype> *net = this->GetNet();
    int iter = net->iter();

    int size=height*width*channels;
    for(int n=0; n<num; n++)
    {
        char filename[256];
        if(this->layer_param_.writer_param().has_file())
            strcpy(filename,this->layer_param_.writer_param().file().c_str());
        else
        {
            if(num>1)
                sprintf(filename,"%s/%s%07d(%03d)%s.flo",
                    this->layer_param_.writer_param().folder().c_str(),
                    this->layer_param_.writer_param().prefix().c_str(),
                    iter,
                    n,
                    this->layer_param_.writer_param().suffix().c_str()
                );
            else
                sprintf(filename,"%s/%s%07d%s.flo",
                    this->layer_param_.writer_param().folder().c_str(),
                    this->layer_param_.writer_param().prefix().c_str(),
                    iter,
                    this->layer_param_.writer_param().suffix().c_str()
                );
        }

        const Dtype* data=bottom[0]->cpu_data()+n*size;

        LOG(INFO) << "Saving " << filename;
        writeFloFile(filename,(const float*)data,width,height);
    }
}


#ifdef CPU_ONLY
STUB_GPU_FORWARD(FLOWriterLayer, Forward);
#endif

INSTANTIATE_CLASS(FLOWriterLayer);
REGISTER_LAYER_CLASS(FLOWriter);

}  // namespace caffe
