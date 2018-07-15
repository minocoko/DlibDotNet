#ifndef _CPP_FACE_RECOGNITION_H_
#define _CPP_FACE_RECOGNITION_H_

// Copyright (C) 2017  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "../export.h"
#include "face_recognition_model_v1.h"
#include <dlib/matrix.h>
#include <dlib/geometry/vector.h>
#include <dlib/dnn.h>
#include <dlib/image_transforms.h>
#include <dlib/image_io.h>
#include <dlib/clustering.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/pixel.h>
#include "../shared.h"

using namespace dlib;
using namespace std;

namespace dlib {
	namespace tools {

		class face_recognition
		{

		public:
			face_recognition() {

			}

			face_recognition(
				const std::string& cnn_face_detection_model_filename,
				const std::string& recognition_model_filename,
				const std::string& predictor_model_filename)
			{
				deserialize(recognition_model_filename) >> net;
				deserialize(predictor_model_filename) >> sp;

				face_detector = get_frontal_face_detector();
				cnn_face_detector = face_recognition_model_v1(cnn_face_detection_model_filename);
			}

			template <typename image_type>
			std::vector<rectangle> face_locations(
				const image_type& img,
				unsigned int number_of_times_to_upsample = 0,
				const std::string& model = "hog") {
				if (model == "cnn") {
					std::vector<mmod_rect> result = cnn_face_detector.detect(img, number_of_times_to_upsample);
					std::vector<rectangle> output;
					for (int index = 0; index < result.size(); index++) {
						mmod_rect& mmod_det = result[index];
						output.push_back(move(mmod_det.rect));
					}

					return output;
				}
				else {
					return face_detector(img, number_of_times_to_upsample);
				}
			}

			template <typename image_type>
			std::vector<full_object_detection> face_landmarks(
				const image_type& img,
				const std::vector<rectangle>& rectangles) {
				std::vector<full_object_detection> faces;
				for (int index = 0; index < rectangles.size(); index++) {
					auto shape = sp(img, rectangles[index]);
					faces.push_back(shape);
				}

				return faces;
			}

			template <typename image_type>
			std::vector<matrix<float, 0, 1>> face_encodings(
				const image_type& img,
				const std::vector<full_object_detection>& landmarks,
				const double padding = 0.2) {
				std::vector<matrix<rgb_pixel>> faces;
				for (int index = 0; index < landmarks.size(); index++) {
					matrix<rgb_pixel> face_chip;
					extract_image_chip(img, get_face_chip_details(landmarks[index], input_rgb_image_size, padding), face_chip);
					faces.push_back(move(face_chip));
				}
				return net(faces);
			}

			template <typename image_type>
			std::vector<matrix<float, 0, 1>> face_encodings(
				const image_type& img,
				const std::vector<rectangle>& rectangles,
				const double padding = 0.2) {
				std::vector<matrix<rgb_pixel>> faces;
				for (int index = 0; index < rectangles.size(); index++) {
					auto shape = sp(img, rectangles[index]);
					matrix<rgb_pixel> face_chip;
					extract_image_chip(img, get_face_chip_details(shape, input_rgb_image_size, padding), face_chip);
					faces.push_back(move(face_chip));
				}
				return net(faces);
			}

			bool face_compare(matrix<float, 0, 1> known_face_encodings, matrix<float, 0, 1> face_encoding_to_check, float tolerance = 0.6) {
				return length(known_face_encodings - face_encoding_to_check) < tolerance;
			}

		private:
			static const int input_rgb_image_size = 150;
			shape_predictor sp;
			dlib::rand rnd;

			std::vector<array2d<rgb_pixel>> jitter_image(
				const array2d<rgb_pixel>& img,
				const int num_jitters
			)
			{
				std::vector<array2d<rgb_pixel>> crops;
				for (int i = 0; i < num_jitters; ++i)
					crops.push_back(dlib::jitter_image(img, rnd));
				return crops;
			}


			template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
			using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

			template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
			using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

			template <int N, template <typename> class BN, int stride, typename SUBNET>
			using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

			template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
			template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

			template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
			template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
			template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
			template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
			template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

			using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
				alevel0<
				alevel1<
				alevel2<
				alevel3<
				alevel4<
				max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
				input_rgb_image_sized<input_rgb_image_size>
				>>>>>>>>>>>>;
			anet_type net;

			frontal_face_detector face_detector;
			face_recognition_model_v1 cnn_face_detector;
		};

		// ----------------------------------------------------------------------------------------
		DLLEXPORT face_recognition* face_recognition_new(
			const char* cnn_face_detection_model_filename,
			const char*  recognition_model_filename,
			const char* predictor_model_filename) {
			return new face_recognition(
				std::string(cnn_face_detection_model_filename),
				std::string(recognition_model_filename),
				std::string(predictor_model_filename));
		};

		DLLEXPORT int face_recognition_face_locations(
			std::vector<rectangle*>* output,
			face_recognition* recognitor,
			array2d_type img_type,
			void* img,
			unsigned int number_of_times_to_upsample = 0,
			const char* modelType = "hog") {
			int err = ERR_OK;
			std::string& model = std::string(modelType);

			switch (img_type)
			{
			case array2d_type::UInt8:
			{
				auto result = recognitor->face_locations(*((array2d<uint8_t>*)img), number_of_times_to_upsample, model);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new rectangle(result[index]));
			}
			break;
			case array2d_type::UInt16:
			{
				auto result = recognitor->face_locations(*((array2d<uint16_t>*)img), number_of_times_to_upsample, model);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new rectangle(result[index]));
			}
			break;
			case array2d_type::UInt32:
			{
				auto result = recognitor->face_locations(*((array2d<uint32_t>*)img), number_of_times_to_upsample, model);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new rectangle(result[index]));
			}
			break;
			case array2d_type::Int8:
			{
				auto result = recognitor->face_locations(*((array2d<int8_t>*)img), number_of_times_to_upsample, model);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new rectangle(result[index]));
			}
			break;
			case array2d_type::Int16:
			{
				auto result = recognitor->face_locations(*((array2d<int16_t>*)img), number_of_times_to_upsample, model);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new rectangle(result[index]));
			}
			break;
			case array2d_type::Int32:
			{
				auto result = recognitor->face_locations(*((array2d<int32_t>*)img), number_of_times_to_upsample, model);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new rectangle(result[index]));
			}
			break;
			case array2d_type::Float:
			{
				auto result = recognitor->face_locations(*((array2d<float>*)img), number_of_times_to_upsample, model);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new rectangle(result[index]));
			}
			break;
			case array2d_type::Double:
			{
				auto result = recognitor->face_locations(*((array2d<double>*)img), number_of_times_to_upsample, model);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new rectangle(result[index]));
			}
			break;
			case array2d_type::RgbPixel:
			{
				auto result = recognitor->face_locations(*((array2d<rgb_pixel>*)img), number_of_times_to_upsample, model);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new rectangle(result[index]));
			}
			break;
			case array2d_type::HsiPixel:
			{
				auto result = recognitor->face_locations(*((array2d<hsi_pixel>*)img), number_of_times_to_upsample, model);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new rectangle(result[index]));
			}
			break;
			case array2d_type::RgbAlphaPixel:
			default:
				err = ERR_INPUT_ELEMENT_TYPE_NOT_SUPPORT;
				break;
			}

			return err;
		}

		DLLEXPORT int face_recognition_face_landmarks(
			std::vector<full_object_detection*>* output,
			face_recognition* recognitor,
			array2d_type img_type,
			void* img,
			const std::vector<rectangle*>* rects) {
			int err = ERR_OK;
			auto rectangles = std::vector<rectangle>();
			for (int index = 0; index < rects->size(); index++) {
				rectangles.push_back(*((*rects)[index]));
			}

			switch (img_type)
			{
			case array2d_type::UInt8:
			{
				auto result = recognitor->face_landmarks(*((array2d<uint8_t>*)img), rectangles);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new full_object_detection(result[index]));
			}
			break;
			case array2d_type::UInt16:
			{
				auto result = recognitor->face_landmarks(*((array2d<uint16_t>*)img), rectangles);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new full_object_detection(result[index]));
			}
			break;
			case array2d_type::UInt32:
			{
				auto result = recognitor->face_landmarks(*((array2d<uint32_t>*)img), rectangles);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new full_object_detection(result[index]));
			}
			break;
			case array2d_type::Int8:
			{
				auto result = recognitor->face_landmarks(*((array2d<int8_t>*)img), rectangles);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new full_object_detection(result[index]));
			}
			break;
			case array2d_type::Int16:
			{
				auto result = recognitor->face_landmarks(*((array2d<int16_t>*)img), rectangles);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new full_object_detection(result[index]));
			}
			break;
			case array2d_type::Int32:
			{
				auto result = recognitor->face_landmarks(*((array2d<int32_t>*)img), rectangles);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new full_object_detection(result[index]));
			}
			break;
			case array2d_type::Float:
			{
				auto result = recognitor->face_landmarks(*((array2d<float>*)img), rectangles);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new full_object_detection(result[index]));
			}
			break;
			case array2d_type::Double:
			{
				auto result = recognitor->face_landmarks(*((array2d<double>*)img), rectangles);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new full_object_detection(result[index]));
			}
			break;
			case array2d_type::RgbPixel:
			{
				auto result = recognitor->face_landmarks(*((array2d<rgb_pixel>*)img), rectangles);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new full_object_detection(result[index]));
			}
			break;
			case array2d_type::HsiPixel:
			{
				auto result = recognitor->face_landmarks(*((array2d<hsi_pixel>*)img), rectangles);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new full_object_detection(result[index]));
			}
			break;
			case array2d_type::RgbAlphaPixel:
			default:
				err = ERR_INPUT_ELEMENT_TYPE_NOT_SUPPORT;
				break;
			}

			return err;
		}

		DLLEXPORT int face_recognition_face_encodings(
			std::vector<matrix<float, 0, 1>*>* output,
			face_recognition* recognitor,
			array2d_type img_type,
			void* img,
			const std::vector<rectangle*>* rects,

			const double padding = 0.2) {
			int err = ERR_OK;
			auto rectangles = std::vector<rectangle>();
			for (int index = 0; index < rects->size(); index++) {
				rectangles.push_back(*((*rects)[index]));
			}

			switch (img_type)
			{
			case array2d_type::UInt8:
			{
				auto result = recognitor->face_encodings(*((array2d<uint8_t>*)img), rectangles, padding);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new matrix<float, 0, 1>(result[index]));
			}
			break;
			case array2d_type::UInt16:
			{
				auto result = recognitor->face_encodings(*((array2d<uint16_t>*)img), rectangles, padding);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new matrix<float, 0, 1>(result[index]));
			}
			break;
			case array2d_type::UInt32:
			{
				auto result = recognitor->face_encodings(*((array2d<uint32_t>*)img), rectangles, padding);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new matrix<float, 0, 1>(result[index]));
			}
			break;
			case array2d_type::Int8:
			{
				auto result = recognitor->face_encodings(*((array2d<int8_t>*)img), rectangles, padding);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new matrix<float, 0, 1>(result[index]));
			}
			break;
			case array2d_type::Int16:
			{
				auto result = recognitor->face_encodings(*((array2d<int16_t>*)img), rectangles, padding);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new matrix<float, 0, 1>(result[index]));
			}
			break;
			case array2d_type::Int32:
			{
				auto result = recognitor->face_encodings(*((array2d<int32_t>*)img), rectangles, padding);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new matrix<float, 0, 1>(result[index]));
			}
			break;
			case array2d_type::Float:
			{
				auto result = recognitor->face_encodings(*((array2d<float>*)img), rectangles, padding);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new matrix<float, 0, 1>(result[index]));
			}
			break;
			case array2d_type::Double:
			{
				auto result = recognitor->face_encodings(*((array2d<double>*)img), rectangles, padding);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new matrix<float, 0, 1>(result[index]));
			}
			break;
			case array2d_type::RgbPixel:
			{
				auto result = recognitor->face_encodings(*((array2d<rgb_pixel>*)img), rectangles, padding);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new matrix<float, 0, 1>(result[index]));
			}
			break;
			case array2d_type::HsiPixel:
			{
				auto result = recognitor->face_encodings(*((array2d<hsi_pixel>*)img), rectangles, padding);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new matrix<float, 0, 1>(result[index]));
			}
			break;
			case array2d_type::RgbAlphaPixel:
			default:
				err = ERR_INPUT_ELEMENT_TYPE_NOT_SUPPORT;
				break;
			}

			return err;
		};

		DLLEXPORT int face_recognition_face_encodings2(
			std::vector<matrix<float, 0, 1>*>* output,
			face_recognition* recognitor,
			array2d_type img_type,
			void* img,
			const std::vector<full_object_detection*>* rects,
			const double padding = 0.2) {
			int err = ERR_OK;
			auto landmarks = std::vector<full_object_detection>();
			for (int index = 0; index < rects->size(); index++) {
				landmarks.push_back(*((*rects)[index]));
			}

			switch (img_type)
			{
			case array2d_type::UInt8:
			{
				auto result = recognitor->face_encodings(*((array2d<uint8_t>*)img), landmarks, padding);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new matrix<float, 0, 1>(result[index]));
			}
			break;
			case array2d_type::UInt16:
			{
				auto result = recognitor->face_encodings(*((array2d<uint16_t>*)img), landmarks, padding);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new matrix<float, 0, 1>(result[index]));
			}
			break;
			case array2d_type::UInt32:
			{
				auto result = recognitor->face_encodings(*((array2d<uint32_t>*)img), landmarks, padding);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new matrix<float, 0, 1>(result[index]));
			}
			break;
			case array2d_type::Int8:
			{
				auto result = recognitor->face_encodings(*((array2d<int8_t>*)img), landmarks, padding);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new matrix<float, 0, 1>(result[index]));
			}
			break;
			case array2d_type::Int16:
			{
				auto result = recognitor->face_encodings(*((array2d<int16_t>*)img), landmarks, padding);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new matrix<float, 0, 1>(result[index]));
			}
			break;
			case array2d_type::Int32:
			{
				auto result = recognitor->face_encodings(*((array2d<int32_t>*)img), landmarks, padding);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new matrix<float, 0, 1>(result[index]));
			}
			break;
			case array2d_type::Float:
			{
				auto result = recognitor->face_encodings(*((array2d<float>*)img), landmarks, padding);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new matrix<float, 0, 1>(result[index]));
			}
			break;
			case array2d_type::Double:
			{
				auto result = recognitor->face_encodings(*((array2d<double>*)img), landmarks, padding);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new matrix<float, 0, 1>(result[index]));
			}
			break;
			case array2d_type::RgbPixel:
			{
				auto result = recognitor->face_encodings(*((array2d<rgb_pixel>*)img), landmarks, padding);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new matrix<float, 0, 1>(result[index]));
			}
			break;
			case array2d_type::HsiPixel:
			{
				auto result = recognitor->face_encodings(*((array2d<hsi_pixel>*)img), landmarks, padding);
				for (int index = 0; index < result.size(); index++)
					output->push_back(new matrix<float, 0, 1>(result[index]));
			}
			break;
			case array2d_type::RgbAlphaPixel:
			default:
				err = ERR_INPUT_ELEMENT_TYPE_NOT_SUPPORT;
				break;
			}

			return err;
		};

		DLLEXPORT bool face_recognition_face_compare(
			face_recognition* recognitor,
			matrix<float, 0, 1>* known_face_encodings,
			matrix<float, 0, 1>* face_encoding_to_check,
			float tolerance = 0.6) {
			return recognitor->face_compare(*known_face_encodings, *face_encoding_to_check, tolerance);
		};

		DLLEXPORT void face_recognition_delete(face_recognition* obj) {
			delete obj;
		};
	}
}

#endif
