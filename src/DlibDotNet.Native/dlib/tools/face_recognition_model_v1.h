#ifndef _CPP_FACE_RECOGNITION_MODEL_V1_H_
#define _CPP_FACE_RECOGNITION_MODEL_V1_H_

// Copyright (C) 2017  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "../export.h"
#include <dlib/matrix.h>
#include <dlib/array2d.h>
#include <dlib/dnn.h>
#include <dlib/image_transforms.h>
#include <dlib/pixel.h>
#include "../shared.h"

using namespace dlib;
using namespace std;

namespace dlib {
	namespace tools {
		class face_recognition_model_v1
		{

		public:

			face_recognition_model_v1() {

			}

			face_recognition_model_v1(const std::string& model_filename)
			{
				deserialize(model_filename) >> net;
			}

			template <typename image_type>
			std::vector<mmod_rect> detect(
				const image_type& img,
				const int upsample_num_times
			)
			{
				pyramid_down<2> pyr;
				std::vector<mmod_rect> rects;

				matrix<rgb_pixel>& newImg = (matrix<rgb_pixel>&)img;

				// Copy the data into dlib based objects
				matrix<rgb_pixel> image;
				//if (is_image<unsigned char>(pyimage))
				//    assign_image(image, numpy_image<unsigned char>(pyimage));
				//else if (is_image<rgb_pixel>(pyimage))
				assign_image(image, newImg);
				//else
				//    throw dlib::error("Unsupported image type, must be 8bit gray or RGB image.");

				// Upsampling the image will allow us to detect smaller faces but will cause the
				// program to use more RAM and run longer.
				unsigned int levels = upsample_num_times;
				while (levels > 0)
				{
					levels--;
					pyramid_up(image, pyr);
				}

				auto dets = net(image);

				// Scale the detection locations back to the original image size
				// if the image was upscaled.
				for (auto&& d : dets) {
					d.rect = pyr.rect_down(d.rect, upsample_num_times);
					rects.push_back(d);
				}

				return rects;
			}

			template <typename image_type>
			std::vector<std::vector<mmod_rect>> detect_mult(
				const std::vector<image_type>& imgs,
				const int upsample_num_times,
				const int batch_size = 128
			)
			{
				pyramid_down<2> pyr;
				std::vector<matrix<rgb_pixel>> dimgs;
				dimgs.reserve(imgs.size());

				for (int i = 0; i < dimgs.size(); i++)
				{
					// Copy the data into dlib based objects
					matrix<rgb_pixel> image;
					//const image_type& img = imgs[i];
					matrix<rgb_pixel>& newImg = (matrix<rgb_pixel>&)imgs[i];
					assign_image(image, newImg);
					//py::array tmp = imgs[i].cast<py::array>();
					//if (is_image<unsigned newImg>(tmp))
					//    assign_image(image, numpy_image<unsigned char>(tmp));
					//else if (is_image<rgb_pixel>(tmp))
					//    assign_image(image, numpy_image<rgb_pixel>(tmp));
					//else
					//    throw dlib::error("Unsupported image type, must be 8bit gray or RGB image.");

					for (int i = 0; i < upsample_num_times; i++)
					{
						pyramid_up(image);
					}
					dimgs.emplace_back(std::move(image));
				}

				for (int i = 1; i < dimgs.size(); i++)
				{
					if (dimgs[i - 1].nc() != dimgs[i].nc() || dimgs[i - 1].nr() != dimgs[i].nr())
						throw dlib::error("Images in list must all have the same dimensions.");

				}

				auto dets = net(dimgs, batch_size);
				std::vector<std::vector<mmod_rect> > all_rects;

				for (auto&& im_dets : dets)
				{
					std::vector<mmod_rect> rects;
					rects.reserve(im_dets.size());
					for (auto&& d : im_dets) {
						d.rect = pyr.rect_down(d.rect, upsample_num_times);
						rects.push_back(d);
					}
					all_rects.push_back(rects);
				}

				return all_rects;
			}

		private:

			template <long num_filters, typename SUBNET> using con5d = con<num_filters, 5, 5, 2, 2, SUBNET>;
			template <long num_filters, typename SUBNET> using con5 = con<num_filters, 5, 5, 1, 1, SUBNET>;

			template <typename SUBNET> using downsampler = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16, SUBNET>>>>>>>>>;
			template <typename SUBNET> using rcon5 = relu<affine<con5<45, SUBNET>>>;

			using net_type = loss_mmod<con<1, 9, 9, 1, 1, rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

			net_type net;
		};

		// ----------------------------------------------------------------------------------------
		//DLLEXPORT face_recognition_model_v1* face_recognition_model_v1_new(const char*  model_filename) {
		//	std::string file_name = std::string(model_filename);
		//	return new face_recognition_model_v1(file_name);
		//};

		//DLLEXPORT int face_recognition_model_v1_detect(
		//	face_recognition_model_v1* detector,
		//	std::vector<mmod_rect*>* dets,
		//	array2d_type img_type,
		//	void* img,
		//	const int upsample_num_times) {
		//	int err = ERR_OK;
		//	std::vector<mmod_rect>* result = NULL;

		//	switch (img_type)
		//	{
		//	case array2d_type::UInt8:
		//	{
		//		result = &detector->detect(*((array2d<uint8_t>*)img), upsample_num_times);
		//	}
		//	break;
		//	case array2d_type::UInt16:
		//	{
		//		result = &detector->detect(*((array2d<uint16_t>*)img), upsample_num_times);
		//	}
		//	break;
		//	case array2d_type::UInt32:
		//	{
		//		result = &detector->detect(*((array2d<uint32_t>*)img), upsample_num_times);
		//	}
		//	break;
		//	case array2d_type::Int8:
		//	{
		//		result = &detector->detect(*((array2d<int8_t>*)img), upsample_num_times);
		//	}
		//	break;
		//	case array2d_type::Int16:
		//	{
		//		result = &detector->detect(*((array2d<int16_t>*)img), upsample_num_times);
		//	}
		//	break;
		//	case array2d_type::Int32:
		//	{
		//		result = &detector->detect(*((array2d<int32_t>*)img), upsample_num_times);
		//	}
		//	break;
		//	case array2d_type::Float:
		//	{
		//		result = &detector->detect(*((array2d<float>*)img), upsample_num_times);
		//	}
		//	break;
		//	case array2d_type::Double:
		//	{
		//		result = &detector->detect(*((array2d<double>*)img), upsample_num_times);
		//	}
		//	break;
		//	case array2d_type::RgbPixel:
		//	{
		//		result = &detector->detect(*((array2d<rgb_pixel>*)img), upsample_num_times);
		//	}
		//	break;
		//	case array2d_type::HsiPixel:
		//	{
		//		result = &detector->detect(*((array2d<hsi_pixel>*)img), upsample_num_times);
		//	}
		//	break;
		//	case array2d_type::RgbAlphaPixel:
		//	default:
		//		err = ERR_INPUT_ELEMENT_TYPE_NOT_SUPPORT;
		//		break;
		//	}

		//	if (NULL != result) {
		//		for (int index = 0; index < result->size(); index++)
		//			dets->push_back(new mmod_rect((*result)[index]));
		//	}

		//	return err;
		//};

		//DLLEXPORT int face_recognition_model_v1_detect_mult(
		//	face_recognition_model_v1* detector,
		//	std::vector<std::vector<mmod_rect>*>* dets,
		//	std::vector<array2d<rgb_pixel>> dimgs,
		//	const int upsample_num_times) {
		//	int err = ERR_OK;
		//	auto result = detector->detect_mult(dimgs, upsample_num_times);
		//	for (int index = 0; index < result.size(); index++)
		//		dets->push_back(new std::vector<mmod_rect>(result[index]));

		//	return err;
		//};

		//DLLEXPORT void face_recognition_model_v1_delete(face_recognition_model_v1* obj) {
		//	delete obj;
		//};
	}
}

#endif
