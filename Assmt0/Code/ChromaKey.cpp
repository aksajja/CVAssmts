#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

namespace {

// Convert BGR color to YCrCb color space.

cv::Scalar bgr2ycrcb( cv::Scalar bgr )
{
	double R = bgr[ 2 ];
	double G = bgr[ 1 ];
	double B = bgr[ 0 ];
	double delta = 128; 

	double Y  = 16 + (65.481 * R + 128.553 * G + 24.966 * B) / 255;
	double Cr = delta + (112.0 * R - 93.786 * G - 18.214 * B) / 255;
	double Cb = delta - (37.797 * R + 74.203 * G - 112.0 * B) / 255;

	return cv::Scalar( Y, Cr, Cb, 0  );
}

/**
 * @param imageBGR   Image with monochrome background
 * @param chromaBGR  Color of the background (using channel order BGR and range [0, 255])
 * @param tInner     Inner threshold, color distances below this value will be counted as foreground
 * @param tOuter     Outer threshold, color distances above this value will be counted as background
 *
 * @return  Mask (0 - background, 255 - foreground, [1, 255] - partially fore- and background)
 */

cv::Mat1b chromaKey( const cv::Mat3b & imageBGR, cv::Scalar chromaBGR, double tInner, double tOuter )
{
	// Basic outline:
	// 1. Convert the image to YCrCb.
	// 2. Measure Euclidean distances of color in YCrBr to chromaKey value.
	// 3. Categorize pixels:
	//   * color distances below inner threshold count as FG; mask value = 255
	//   * color distances above outer threshold count as BG; mask value = 0
	//   * color distances between inner and outer threshold a linearly interpolated; mask value = [0, 255]

	assert( tInner <= tOuter );

	// Convert to YCrCb.
	assert( ! imageBGR.empty() );
	cv::Size imageSize = imageBGR.size();
	cv::Mat3b imageYCrCb;
	cv::cvtColor( imageBGR, imageYCrCb, cv::COLOR_BGR2YCrCb );
	cv::Scalar chromaYCrCb = bgr2ycrcb( chromaBGR ); // Convert a single BGR value to YCrCb.

	// Build the mask.
	cv::Mat1b mask = cv::Mat1b::zeros( imageSize );
	const cv::Vec3d key( chromaYCrCb[ 0 ], chromaYCrCb[ 1 ], chromaYCrCb[ 2 ] );
	//const cv::Vec3d key(41, 110, 240); 

	std::cout << chromaYCrCb[0] << " " << chromaYCrCb[1] << " " << chromaYCrCb[2] <<std::endl; 
	for ( int y = 0; y < imageSize.height; ++y )
	{
		for ( int x = 0; x < imageSize.width; ++x )
		{
			const cv::Vec3d color( imageYCrCb( y, x )[ 0 ], imageYCrCb( y, x )[ 1 ], imageYCrCb( y, x )[ 2 ] );
			double distance = cv::norm( key - color );

			if ( distance < tInner )
			{
				// Current pixel is fully part of the background.
				mask( y, x ) = 0;
			}
			else if ( distance > tOuter )
			{
				// Current pixel is fully part of the foreground.
				mask( y, x ) = 255;
			}
			else
			{
				// Current pixel is partially part both, fore- and background; interpolate linearly.
				// Compute the interpolation factor and clip its value to the range [0, 255].
				double d1 = distance - tInner;
				double d2 = tOuter   - tInner;
				uint8_t alpha = static_cast< uint8_t >( 255. * ( d1 / d2 ) );

				mask( y, x ) = alpha;
			}
		}
	}

	return mask;
}

cv::Mat3b replaceBackground( const cv::Mat3b & image, const cv::Mat1b & mask, const cv::Mat3b & image_bg )
{
	cv::Size imageSize = image.size();
	cv::Mat3b newImage( image.size() );

	for ( int y = 0; y < imageSize.height; ++y )
	{
		for ( int x = 0; x < imageSize.width; ++x )
		{
			uint8_t maskValue = mask( y, x );

			if ( maskValue >= 255 )
			{
				newImage( y, x ) = image( y, x );
			}
			else if ( maskValue <= 0 )
			{
				newImage( y, x ) = image_bg( y, x);
			}
			else
			{
				double alpha = 128. / static_cast< double >( maskValue );
				newImage( y, x ) = alpha * image( y, x ) + ( 1. - alpha ) * image_bg( y, x);
			}
		}
	}

	return newImage;
}

} // namespace

int main()
{

	for (int i=1; i<24; i++){

		std::string inputFilename = "./tiger_out_cropped/cropped_"+std::to_string(i)+".jpg";
		std::string maskFilename = "./mask.jpg";
		std::string newBackgroundFilename = "./out_images/bg_"+std::to_string(i)+".jpg";
		std::string outputFilename = "./compositeVideo/newBG_"+std::to_string(i)+".jpg";

		// Load the input image.
		cv::Mat3b input = cv::imread( inputFilename, cv::IMREAD_COLOR );
		cv::Mat3b output = cv::imread( newBackgroundFilename, cv::IMREAD_COLOR );

		if ( input.empty() )
		{
			std::cerr << "Input file <" << inputFilename << "> could not be loaded ... " << std::endl;

			return 1;
		}

		// Apply the chroma keying and save the output.
		cv::Scalar chroma( 0, 255, 0, 0 );
		//cv::Scalar chroma( 255, 0, 0, 0 );
		std::cout<<"chroma is "<< chroma[0] << chroma[1] << chroma[2] << chroma[3]<<std::endl;
		double tInner = 100.;
		double tOuter = 170.;
		cv::Mat1b mask = chromaKey( input, chroma, tInner, tOuter );

		cv::Mat3b newBackground = replaceBackground( input, mask, output );

		cv::imwrite( maskFilename, mask );
		cv::imwrite( outputFilename, newBackground );

	}

	return 0;
}
