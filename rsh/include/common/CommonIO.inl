#include "common/CommonIO.hpp"
#include <sstream>
#include <cstdint>

#include <opencv2/core.hpp>

using namespace std;

inline Intrinsic CommonIO::ReadIntrinsic(const std::string &filename, int width, int height)
{
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	cv::Mat K;
	Intrinsic result;

	if (!fs.isOpened()) {
		throw runtime_error("!![Error] CommonIO: Can't open " + filename);
	}

	fs["M"] >> K;
	K.convertTo(K, CV_64FC1);
	result = Intrinsic(K.at<double>(2), K.at<double>(5), K.at<double>(0),
		K.at<double>(4), width, height);

	return result;
}

inline Extrinsic CommonIO::ReadExtrinsic(const std::string &filename)
{
	cv::FileStorage file(filename, cv::FileStorage::READ);
	cv::Mat R, T;
	Extrinsic result;
	glm::vec3 pos, target, up;

	if (!file.isOpened()) {
		throw runtime_error("!![Error] CommonIO: Can't open " + filename);
	}

	/* R, T are stored in the form of world2cam */
	file["R"] >> R;
	T.convertTo(T, CV_64FC1);
	file["T"] >> T;
	R.convertTo(R, CV_64FC1);
	T = -R.inv() * T;		// to get the right camera position
	R = R.inv();			// to get the right up/dir direction

	pos = glm::vec3(T.at<double>(0), T.at<double>(1), T.at<double>(2));
	up = -glm::vec3(R.at<double>(1), R.at<double>(4), R.at<double>(7));
	target = pos + glm::vec3(R.at<double>(2), R.at<double>(5), R.at<double>(8));
	result = Extrinsic(pos, target, up);

	return result;
}
