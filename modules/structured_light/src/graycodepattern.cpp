/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2015, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#include "precomp.hpp"
#include <numeric>

namespace cv {
namespace structured_light {
class CV_EXPORTS_W GrayCodePattern_Impl CV_FINAL : public GrayCodePattern
{
 public:
  // Constructor
  explicit GrayCodePattern_Impl( const GrayCodePattern::Params &parameters = GrayCodePattern::Params() );

  // Destructor
  virtual ~GrayCodePattern_Impl() CV_OVERRIDE {};

  // Generates the gray code pattern as a std::vector<Mat>
  bool generate( OutputArrayOfArrays patternImages ) CV_OVERRIDE;

  // Decodes the gray code pattern, computing the disparity map
  bool decode( const std::vector< std::vector<Mat> >& patternImages, OutputArray disparityMap, 
                       OutputArrayOfArrays shadowMasks = noArray(), InputArrayOfArrays blackImages = noArray(),
               InputArrayOfArrays whiteImages = noArray(), int flags = DECODE_3D_UNDERWORLD ) const CV_OVERRIDE;

  // Returns the number of pattern images for the graycode pattern
  size_t getNumberOfPatternImages() const CV_OVERRIDE;

  // Sets the value for black threshold
  void setBlackThreshold( size_t val ) CV_OVERRIDE;

  // Sets the value for set the value for white threshold
  void setWhiteThreshold( size_t val ) CV_OVERRIDE;

  // Generates the images needed for shadowMasks computation
  void getImagesForShadowMasks( InputOutputArray blackImage, InputOutputArray whiteImage ) const CV_OVERRIDE;

  bool getProjPixel(InputArrayOfArrays patternImages, int x, int y, CV_OUT Point& projPix) const CV_OVERRIDE;

 private:
  // Parameters
  Params params;

  // The number of images of the pattern
  size_t numOfPatternImages;

  // The number of row images of the pattern
  size_t numOfRowImgs;

  // The number of column images of the pattern
  size_t numOfColImgs;

  // Number between 0-255 that represents the minimum brightness difference
  // between the fully illuminated (white) and the non - illuminated images (black)
  size_t blackThreshold;

  // Number between 0-255 that represents the minimum brightness difference
  // between the gray-code pattern and its inverse images
  size_t whiteThreshold;

  // Computes the required number of pattern images, allocating the pattern vector
  void computeNumberOfPatternImages();

  // Computes the shadows occlusion where we cannot reconstruct the model
  void computeShadowMasks( InputArrayOfArrays blackImages, InputArrayOfArrays whiteImages,
                           OutputArrayOfArrays shadowMasks ) const;

  bool getProjPixelFast(const std::vector<Mat>& fastColImages, const std::vector<Mat>& fastRowImages, int x, int y, Point& projPix) const;
  void populateFastPatternImages(const std::vector<std::vector<Mat>>& acquiredPatterns, std::vector<std::vector<Mat>>& fastPatternColImages, std::vector<std::vector<Mat>>& fastPatternRowImages) const;
};

/*
 *  GrayCodePattern
 */
GrayCodePattern::Params::Params()
{
  width = 1024;
  height = 768;
}

GrayCodePattern_Impl::GrayCodePattern_Impl( const GrayCodePattern::Params &parameters ) :
    params( parameters )
{
  computeNumberOfPatternImages();
  blackThreshold = 40;  // 3D_underworld default value
  whiteThreshold = 5;   // 3D_underworld default value
}

bool GrayCodePattern_Impl::generate( OutputArrayOfArrays pattern )
{
  std::vector<Mat>& pattern_ = *( std::vector<Mat>* ) pattern.getObj();
  pattern_.resize( numOfPatternImages );

  for( size_t i = 0; i < numOfPatternImages; i++ )
  {
    pattern_[i] = Mat( params.height, params.width, CV_8U );
  }

  uchar flag = 0;

  for( int j = 0; j < params.width; j++ )  // rows loop
  {
    int rem = 0, num = j, prevRem = j % 2;

    for( size_t k = 0; k < numOfColImgs; k++ )  // images loop
    {
      num = num / 2;
      rem = num % 2;

      if( ( rem == 0 && prevRem == 1 ) || ( rem == 1 && prevRem == 0) )
      {
        flag = 1;
      }
      else
      {
        flag = 0;
      }

      for( int i = 0; i < params.height; i++ )  // rows loop
      {

        uchar pixel_color = ( uchar ) flag * 255;

        pattern_[2 * numOfColImgs - 2 * k - 2].at<uchar>( i, j ) = pixel_color;
        if( pixel_color > 0 )
          pixel_color = ( uchar ) 0;
        else
         pixel_color = ( uchar ) 255;
        pattern_[2 * numOfColImgs - 2 * k - 1].at<uchar>( i, j ) = pixel_color;  // inverse
      }

      prevRem = rem;
    }
  }

  for( int i = 0; i < params.height; i++ )  // rows loop
  {
    int rem = 0, num = i, prevRem = i % 2;

    for( size_t k = 0; k < numOfRowImgs; k++ )
    {
      num = num / 2;
      rem = num % 2;

      if( (rem == 0 && prevRem == 1) || (rem == 1 && prevRem == 0) )
      {
        flag = 1;
      }
      else
      {
        flag = 0;
      }

      for( int j = 0; j < params.width; j++ )
      {

        uchar pixel_color = ( uchar ) flag * 255;
        pattern_[2 * numOfRowImgs - 2 * k + 2 * numOfColImgs - 2].at<uchar>( i, j ) = pixel_color;

        if( pixel_color > 0 )
          pixel_color = ( uchar ) 0;
        else
          pixel_color = ( uchar ) 255;

        pattern_[2 * numOfRowImgs - 2 * k + 2 * numOfColImgs - 1].at<uchar>( i, j ) = pixel_color;
      }

      prevRem = rem;
    }
  }

  return true;
}


bool GrayCodePattern_Impl::decode(const std::vector< std::vector<Mat> >& patternImages, OutputArray disparityMap,
    OutputArrayOfArrays outShadowMasks,
    InputArrayOfArrays blackImages, InputArrayOfArrays whiteImages, int flags) const
{
    const std::vector<std::vector<Mat>>& acquired_pattern = patternImages;

    if (flags == DECODE_3D_UNDERWORLD)
    {
        // Shadow mask computation remains the same
        std::vector<Mat> shadowMasks;
        computeShadowMasks(blackImages, whiteImages, shadowMasks);

        std::vector<Mat>& shadowMasks_ = *(std::vector<Mat>*) outShadowMasks.getObj();
        shadowMasks_.resize(shadowMasks.size() + 1);
        for (size_t i = 0; i < shadowMasks.size(); i++)
        {
            shadowMasks_[i] = shadowMasks[i].clone();
        }

        int num_cameras = static_cast<int>(acquired_pattern.size());
        CV_Assert(num_cameras == 2 && "This implementation expects a stereo camera setup (2 cameras).");

        int cam_width = acquired_pattern[0][0].cols;
        int cam_height = acquired_pattern[0][0].rows;
        int proj_width = params.width;
        int proj_height = params.height;

        std::vector<std::vector<Mat>> fastPatternColImages;
        std::vector<std::vector<Mat>> fastPatternRowImages;
        populateFastPatternImages(acquired_pattern, fastPatternColImages, fastPatternRowImages);

        std::vector<Mat> projectorCoordinateMap(num_cameras);
        parallel_for_(Range(0, num_cameras), [&](const Range& range) {
            for (int k = range.start; k < range.end; k++) {
                projectorCoordinateMap[k] = Mat(cam_height, cam_width, CV_32SC2, Vec2i(-1, -1));
                Point projPixel;
                const auto& fastCol = fastPatternColImages[k];
                const auto& fastRow = fastPatternRowImages[k];
                for (int j = 0; j < cam_height; j++) {
                    for (int i = 0; i < cam_width; i++) {
                        if (shadowMasks[k].at<uchar>(j, i)) {
                            if (!getProjPixelFast(fastCol, fastRow, i, j, projPixel)) {
                                projectorCoordinateMap[k].at<Vec2i>(j, i) = Vec2i(projPixel.x, projPixel.y);
                            }
                        }
                    }
                }
            }
        });

        std::vector<Mat> sumX(num_cameras);
        std::vector<Mat> counts(num_cameras);
        for (int k = 0; k < num_cameras; ++k) {
            sumX[k] = Mat::zeros(proj_height, proj_width, CV_64F);
            counts[k] = Mat::zeros(proj_height, proj_width, CV_32S);
        }
        for (int k = 0; k < num_cameras; ++k) {
            for (int j = 0; j < cam_height; ++j) {
                for (int i = 0; i < cam_width; ++i) {
                    const Vec2i& projPt = projectorCoordinateMap[k].at<Vec2i>(j, i);
                    if (projPt[0] != -1) {
                        sumX[k].at<double>(projPt[1], projPt[0]) += i;
                        counts[k].at<int>(projPt[1], projPt[0]) += 1;
                    }
                }
            }
        }

        Mat counts_64F[2], avgX[2];
        counts[0].convertTo(counts_64F[0], CV_64F);
        counts[1].convertTo(counts_64F[1], CV_64F);
        cv::divide(sumX[0], counts_64F[0], avgX[0]);
        cv::divide(sumX[1], counts_64F[1], avgX[1]);

        Mat projectorDisparity = avgX[1] - avgX[0];
        Mat validProjectorPixels = (counts[0] > 0) & (counts[1] > 0);

        Mat& disparityMap_ = *(Mat*)disparityMap.getObj();
        Mat map_x(cam_height, cam_width, CV_32F);
        Mat map_y(cam_height, cam_width, CV_32F);

        std::vector<Mat> projMapChannels;
        cv::split(projectorCoordinateMap[0], projMapChannels);
        projMapChannels[0].convertTo(map_x, CV_32F); // Projector x-coordinates
        projMapChannels[1].convertTo(map_y, CV_32F); // Projector y-coordinates

        // Remap disparity values. INTER_NEAREST ensures identical results to the loop-based lookup.
        // BORDER_CONSTANT(0) handles pixels with no valid projection, setting their disparity to 0.
        cv::remap(projectorDisparity, disparityMap_, map_x, map_y,
            INTER_NEAREST, BORDER_CONSTANT, Scalar(0));

        // Remap the validity mask and invert it to get the final invalid mask.
        Mat validCameraPixels;
        cv::remap(validProjectorPixels, validCameraPixels, map_x, map_y,
            INTER_NEAREST, BORDER_CONSTANT, Scalar(0));

        Mat invalidMask;
        cv::bitwise_not(validCameraPixels, invalidMask);

        shadowMasks_[shadowMasks.size()] = std::move(invalidMask);
        return true;
    }

    return false;
}

// Computes the required number of pattern images
void GrayCodePattern_Impl::computeNumberOfPatternImages()
{
  numOfColImgs = ( size_t ) ceil( log( double( params.width ) ) / log( 2.0 ) );
  numOfRowImgs = ( size_t ) ceil( log( double( params.height ) ) / log( 2.0 ) );
  numOfPatternImages = 2 * numOfColImgs + 2 * numOfRowImgs;
}

// Returns the number of pattern images to project / decode
size_t GrayCodePattern_Impl::getNumberOfPatternImages() const
{
  return numOfPatternImages;
}

  // Computes the shadows occlusion where we cannot reconstruct the model
void GrayCodePattern_Impl::computeShadowMasks( InputArrayOfArrays blackImages, InputArrayOfArrays whiteImages,
                                                    OutputArrayOfArrays shadowMasks ) const 
{
  std::vector<Mat>& whiteImages_ = *( std::vector<Mat>* ) whiteImages.getObj();
  std::vector<Mat>& blackImages_ = *( std::vector<Mat>* ) blackImages.getObj();
  std::vector<Mat>& shadowMasks_ = *( std::vector<Mat>* ) shadowMasks.getObj();

  shadowMasks_.resize( whiteImages_.size() );

  parallel_for_(Range(0, (int) whiteImages_.size()), [&](const Range& range)
  {
    for( int k = range.start; k < range.end; k++ )
    {
        cv::Mat diffImage;
        cv::absdiff(whiteImages_[k], blackImages_[k], diffImage);
        cv::compare(diffImage, blackThreshold, shadowMasks_[k], cv::CMP_GT);
        shadowMasks_[k] /= 255;
    }
  }); 
}

// Generates the images needed for shadowMasks computation
void GrayCodePattern_Impl::getImagesForShadowMasks( InputOutputArray blackImage, InputOutputArray whiteImage ) const
{
  Mat& blackImage_ = *( Mat* ) blackImage.getObj();
  Mat& whiteImage_ = *( Mat* ) whiteImage.getObj();

  blackImage_ = Mat( params.height, params.width, CV_8U, Scalar( 0 ) );
  whiteImage_ = Mat( params.height, params.width, CV_8U, Scalar( 255 ) );
}

void GrayCodePattern_Impl::populateFastPatternImages(const std::vector<std::vector<Mat>>& acquiredPatterns, std::vector<std::vector<Mat>>& fastPatternColImages, std::vector<std::vector<Mat>>& fastPatternRowImages) const {
    fastPatternColImages.resize(acquiredPatterns.size());
    fastPatternRowImages.resize(acquiredPatterns.size());

    for (size_t i = 0; i < acquiredPatterns.size(); i++) {
        const auto& _patternImages = acquiredPatterns[i];
        auto& fastColImagesN = fastPatternColImages[i]; 
        auto& fastRowImagesN = fastPatternRowImages[i]; 
        fastColImagesN.reserve(numOfColImgs);
        fastRowImagesN.reserve(numOfRowImgs);
        for ( size_t count = 0; count < numOfColImgs; count++ ) {
            cv::Mat diffImage;
            cv::subtract(_patternImages[count * 2], _patternImages[count * 2 + 1], diffImage, cv::noArray(), CV_32S);
            cv::Mat invalidMask = abs(diffImage) < whiteThreshold;
            cv::compare(diffImage, cv::Scalar(0), diffImage, cv::CMP_GT);
            diffImage.setTo(1, invalidMask);
            fastColImagesN.push_back(std::move(diffImage));
        }

        for ( size_t count = 0; count < numOfRowImgs; count++ ) {
            cv::Mat diffImage;
            cv::subtract(_patternImages[count * 2 + numOfColImgs * 2], _patternImages[count * 2 + numOfColImgs * 2 + 1], diffImage, cv::noArray(), CV_32S);
            cv::Mat invalidMask = abs(diffImage) < whiteThreshold;
            cv::compare(diffImage, cv::Scalar(0), diffImage, cv::CMP_GT);
            diffImage.setTo(1, invalidMask);
            fastRowImagesN.push_back(std::move(diffImage));
        }
    }
}

// Converts a gray code sequence (~ binary number) to a decimal number
static int grayToDec(const std::vector<uchar>& gray)
{
    int dec = 0;

    uchar tmp = gray[0];

    if (tmp)
        dec += (int)pow((float)2, int(gray.size() - 1));

    for (int i = 1; i < (int)gray.size(); i++)
    {
        // XOR operation
        tmp = tmp ^ gray[i];
        if (tmp)
            dec += (int)pow((float)2, int(gray.size() - i - 1));
    }

    return dec;
}

bool GrayCodePattern_Impl::getProjPixel(InputArrayOfArrays patternImages, int x, int y, Point& projPix) const {
    std::vector<Mat>& _patternImages = *(std::vector<Mat>*) patternImages.getObj();
    std::vector<uchar> grayCol;
    std::vector<uchar> grayRow;

    bool error = false;
    int xDec, yDec;

    // process column images
    for (size_t count = 0; count < numOfColImgs; count++)
    {
        // get pixel intensity for regular pattern projection and its inverse
        double val1 = _patternImages[count * 2].at<uchar>(Point(x, y));
        double val2 = _patternImages[count * 2 + 1].at<uchar>(Point(x, y));

        // check if the intensity difference between the values of the normal and its inverse projection image is in a valid range
        if (abs(val1 - val2) < whiteThreshold)
            error = true;

        // determine if projection pixel is on or off
        if (val1 > val2)
            grayCol.push_back(1);
        else
            grayCol.push_back(0);
    }

    xDec = grayToDec(grayCol);

    // process row images
    for (size_t count = 0; count < numOfRowImgs; count++)
    {
        // get pixel intensity for regular pattern projection and its inverse
        double val1 = _patternImages[count * 2 + numOfColImgs * 2].at<uchar>(Point(x, y));
        double val2 = _patternImages[count * 2 + numOfColImgs * 2 + 1].at<uchar>(Point(x, y));

        // check if the intensity difference between the values of the normal and its inverse projection image is in a valid range
        if (abs(val1 - val2) < whiteThreshold)
            error = true;

        // determine if projection pixel is on or off
        if (val1 > val2)
            grayRow.push_back(1);
        else
            grayRow.push_back(0);
    }

    yDec = grayToDec(grayRow);

    if ((yDec >= params.height || xDec >= params.width))
    {
        error = true;
    }

    projPix.x = xDec;
    projPix.y = yDec;

    return error;
}

// For a (x,y) pixel of the camera returns the corresponding projector's pixel
bool GrayCodePattern_Impl::getProjPixelFast( const std::vector<Mat>& fastColImages, const std::vector<Mat>& fastRowImages, int x, int y, Point &projPix) const
{
  int xDec = 0, yDec = 0;
  ushort binBit = 0;
  Point pattPoint = Point(x, y);

  // process column images
  for(const auto& fastColImage : fastColImages)
  {
    
    uchar grayBit = fastColImage.at<uchar>(pattPoint);
    // check if the intensity difference between the values of the normal and its inverse projection image is in a valid range
    if( grayBit == 1) return true;

    // determine if projection pixel is on or off
    binBit = binBit ^ (grayBit == 255);
    xDec = (xDec << 1) | binBit;
  }

  binBit = 0; // reset binary bit for row images

  // process row images
  for(const auto& fastRowImage : fastRowImages)
  {
    uchar grayBit = fastRowImage.at<uchar>(pattPoint);
    // check if the intensity difference between the values of the normal and its inverse projection image is in a valid range
    if( grayBit == 1 ) return true;

    // determine if projection pixel is on or off
    binBit = binBit ^ (grayBit == 255);
    yDec = (yDec << 1) | binBit;
  }

  if( (yDec >= params.height || xDec >= params.width) )
  {
    return true;
  }

  projPix.x = xDec;
  projPix.y = yDec;

  return false;
}

// Sets the value for black threshold
void GrayCodePattern_Impl::setBlackThreshold( size_t val )
{
  blackThreshold = val;
}

// Sets the value for white threshold
void GrayCodePattern_Impl::setWhiteThreshold( size_t val )
{
  whiteThreshold = val;
}

// Creates the GrayCodePattern instance
Ptr<GrayCodePattern> GrayCodePattern::create( const GrayCodePattern::Params& params )
{
  return makePtr<GrayCodePattern_Impl>( params );
}

// Creates the GrayCodePattern instance
// alias for scripting
Ptr<GrayCodePattern> GrayCodePattern::create( int width, int height )
{
  Params params;
  params.width = width;
  params.height = height;
  return makePtr<GrayCodePattern_Impl>( params );
}

}
}
