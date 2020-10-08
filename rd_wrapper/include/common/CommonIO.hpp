#ifndef COMMONIO_H
#define COMMONIO_H

#include <fstream>
#include <string>
#include <vector>
#include "camera/Extrinsic.hpp"
#include "camera/Intrinsic.hpp"

class CommonIO
{
public:
	/**
	* Read intrinsic file stored in OpenCV xml format. The intrinsic matrix
	* should be denoted 'M'. Camera size are passed manually.
	*
	* @param filename	The xml file
	* @param width	camera width
	* @param height	camera height
	*
	* @return	The Intrinsic
	*/
	static Intrinsic ReadIntrinsic(const std::string &filename, int width, int height);

	/**
	* Read extrinsic file stored in OpenCV xml format. The file should contain
	* a 3x3 'R' matrix and a 3x1 'T' matrix in world2camera fashion.
	* Note camera y-axis is assumed to align width image coordinates
	*
	* @param filename	The xml file
	*
	* @return	The Extrinsic
	*/
	static Extrinsic ReadExtrinsic(const std::string &);
};

#include "CommonIO.inl"

#endif