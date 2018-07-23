using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

using DlibDotNet;
using DlibDotNet.Extensions;

namespace DlibDotNet.Tools
{
    public class FaceRecognition : DlibObject
    {
        #region Constructors

        public FaceRecognition(string cnnFaceDetectionModelFileName,
                string recognitionModelFileName,
                string predictorModelFileName)
        {
            if (cnnFaceDetectionModelFileName == null)
                throw new ArgumentNullException(nameof(cnnFaceDetectionModelFileName));
            if (recognitionModelFileName == null)
                throw new ArgumentNullException(nameof(recognitionModelFileName));
            if (predictorModelFileName == null)
                throw new ArgumentNullException(nameof(predictorModelFileName));
            if (!File.Exists(cnnFaceDetectionModelFileName))
                throw new FileNotFoundException($"{cnnFaceDetectionModelFileName} is not found", cnnFaceDetectionModelFileName);
            if (!File.Exists(recognitionModelFileName))
                throw new FileNotFoundException($"{recognitionModelFileName} is not found", recognitionModelFileName);
            if (!File.Exists(predictorModelFileName))
                throw new FileNotFoundException($"{predictorModelFileName} is not found", predictorModelFileName);

            this.NativePtr = Native.face_recognition_new(
                Encoding.UTF8.GetBytes(cnnFaceDetectionModelFileName),
                Encoding.UTF8.GetBytes(recognitionModelFileName),
                Encoding.UTF8.GetBytes(predictorModelFileName));
        }

        #endregion

        #region Methods

        public Rectangle[] FaceLocations(Array2DBase image, uint numberOfTimesToUpsample = 0, string model = "hog")
        {
            this.ThrowIfDisposed();

            if (image == null)
                throw new ArgumentNullException(nameof(image));

            image.ThrowIfDisposed();

            var inType = image.ImageType.ToNativeArray2DType();

            using (var dets = new StdVector<Rectangle>())
            {
                var ret = Native.face_recognition_face_locations(dets.NativePtr, this.NativePtr, inType, image.NativePtr, numberOfTimesToUpsample, Encoding.UTF8.GetBytes(model));
                if (ret == Dlib.Native.ErrorType.InputElementTypeNotSupport)
                {
                    throw new ArgumentException($"Input {inType} is not supported.");
                }

                return dets.ToArray();
            }
        }

        public FullObjectDetection[] FaceLandmarks(Array2DBase image, Rectangle[] rects)
        {
            this.ThrowIfDisposed();

            if (image == null)
                throw new ArgumentNullException(nameof(image));
            if (rects == null)
                throw new ArgumentNullException(nameof(rects));

            image.ThrowIfDisposed();

            var inType = image.ImageType.ToNativeArray2DType();
            using (var rectangles = new StdVector<Rectangle>(rects))
            {
                using (var dets = new StdVector<FullObjectDetection>())
                {
                    var ret = Native.face_recognition_face_landmarks(dets.NativePtr, this.NativePtr, inType, image.NativePtr, rectangles.NativePtr);
                    if (ret == Dlib.Native.ErrorType.InputElementTypeNotSupport)
                    {
                        throw new ArgumentException($"Input {inType} is not supported.");
                    }

                    return dets.ToArray();
                }
            }
        }

        public Matrix<double>[] FaceEncodings(Array2DBase image, Rectangle[] rects, double padding = 0.2d)
        {
            this.ThrowIfDisposed();

            if (image == null)
                throw new ArgumentNullException(nameof(image));
            if (rects == null)
                throw new ArgumentNullException(nameof(rects));

            image.ThrowIfDisposed();

            var inType = image.ImageType.ToNativeArray2DType();
            using (var rectangles = new StdVector<Rectangle>(rects))
            {
                using (var dets = new StdVector<Matrix<double>>())
                {
                    var ret = Native.face_recognition_face_encodings(dets.NativePtr, this.NativePtr, inType, image.NativePtr, rectangles.NativePtr, padding);
                    if (ret == Dlib.Native.ErrorType.InputElementTypeNotSupport)
                    {
                        throw new ArgumentException($"Input {inType} is not supported.");
                    }

                    return dets.ToArray();
                }
            }
        }

        public Matrix<double>[] FaceEncodings(Array2DBase image, FullObjectDetection[] landmarks, double padding = 0.2d)
        {
            this.ThrowIfDisposed();

            if (image == null)
                throw new ArgumentNullException(nameof(image));
            if (landmarks == null)
                throw new ArgumentNullException(nameof(landmarks));

            image.ThrowIfDisposed();

            var inType = image.ImageType.ToNativeArray2DType();
            using (var rectangles = new StdVector<FullObjectDetection>((IEnumerable<FullObjectDetection>)landmarks))
            {
                using (var dets = new StdVector<Matrix<double>>())
                {
                    var ret = Native.face_recognition_face_encodings2(dets.NativePtr, this.NativePtr, inType, image.NativePtr, rectangles.NativePtr, padding);
                    if (ret == Dlib.Native.ErrorType.InputElementTypeNotSupport)
                    {
                        throw new ArgumentException($"Input {inType} is not supported.");
                    }

                    return dets.ToArray();
                }
            }
        }

        public bool FaceCompare(Matrix<double> knownFaceCncodings, Matrix<double> faceEncodingToCheck, float tolerance = 0.6f)
        {
            this.ThrowIfDisposed();

            if (knownFaceCncodings == null)
                throw new ArgumentNullException(nameof(knownFaceCncodings));
            if (faceEncodingToCheck == null)
                throw new ArgumentNullException(nameof(faceEncodingToCheck));

            return Native.face_recognition_face_compare(this.NativePtr, knownFaceCncodings.NativePtr, faceEncodingToCheck.NativePtr, tolerance);
        }

        #region Overrides

        protected override void DisposeUnmanaged()
        {
            base.DisposeUnmanaged();
            Native.face_recognition_delete(this.NativePtr);
        }

        #endregion

        #endregion

        internal sealed class Native
        {

            [DllImport(NativeMethods.NativeLibrary, CallingConvention = NativeMethods.CallingConvention)]
            public static extern IntPtr face_recognition_new(
                byte[] cnn_face_detection_model_filename,
                byte[] recognition_model_filename,
                byte[] predictor_model_filename);

            [DllImport(NativeMethods.NativeLibrary, CallingConvention = NativeMethods.CallingConvention)]
            public static extern Dlib.Native.ErrorType face_recognition_face_locations(
                IntPtr output,
                IntPtr recognitor,
                Dlib.Native.Array2DType imgType,
                IntPtr img,
                uint number_of_times_to_upsample,
                byte[] model);

            [DllImport(NativeMethods.NativeLibrary, CallingConvention = NativeMethods.CallingConvention)]
            public static extern Dlib.Native.ErrorType face_recognition_face_landmarks(
                IntPtr output,
                IntPtr recognitor,
                Dlib.Native.Array2DType imgType,
                IntPtr img,
                IntPtr rectangles
                );

            [DllImport(NativeMethods.NativeLibrary, CallingConvention = NativeMethods.CallingConvention)]
            public static extern Dlib.Native.ErrorType face_recognition_face_encodings(
                IntPtr output,
                IntPtr recognitor,
                Dlib.Native.Array2DType imgType,
                IntPtr img,
                IntPtr rectangles,
                double padding = 0.2d);

            [DllImport(NativeMethods.NativeLibrary, CallingConvention = NativeMethods.CallingConvention)]
            public static extern Dlib.Native.ErrorType face_recognition_face_encodings2(
                IntPtr output,
                IntPtr recognitor,
                Dlib.Native.Array2DType imgType,
                IntPtr img,
                IntPtr landmarks,
                double padding = 0.2d);

            [DllImport(NativeMethods.NativeLibrary, CallingConvention = NativeMethods.CallingConvention)]
            public static extern bool face_recognition_face_compare(
                IntPtr recognitor,
                IntPtr known_face_encodings,
                IntPtr face_encoding_to_check,
                float tolerance = 0.6f);

            [DllImport(NativeMethods.NativeLibrary, CallingConvention = NativeMethods.CallingConvention)]
            public static extern void face_recognition_delete(IntPtr point);

        }
    }
}
