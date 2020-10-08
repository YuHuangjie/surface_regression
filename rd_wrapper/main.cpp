#include <iostream>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <glad/glad_egl.h>
#include <glad/glad.h>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include "Renderer.h"
#include "common/CommonIO.hpp"

using namespace std;
namespace fs = boost::filesystem;

namespace {
	// window dimensions
	int CAMERA_WIDTH = 0;
	int CAMERA_HEIGHT = 0;
	const string EXTRINSIC_PATH = "world2depth.xml";
	const string INTRINSIC_PATH = "depth_intrinsic.xml";

	// create opengl context and a window of given size
	int InitGLContext(int window_width, int window_height);
	
	// write image
	void ReadExtrinsicAndIntrinsic(const string &root, vector<string> &e, vector<string> &i,
		vector<string> &sns);
	void PrepareOutputDir(const string &output);
	void WriteImage(unsigned char *data, int width, int height, string filename);
}

int main(int argc, char **argv)
{
	/* Parse command line options */
	namespace po = boost::program_options;
	std::string GeometryFileName = "";
	std::string CameraFolder = "";
	std::string OUTPUT_DIR = "";
	bool save = false;

	// Declare the supported options
	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "display help message")
		("geometry,g", po::value<string>(&GeometryFileName), "geometry file (*.obj)")
		("camera,c", po::value<string>(&CameraFolder), "camera folder containing RT and K")
		("output,o", po::value<string>(&OUTPUT_DIR), "output folder")
		("width,w", po::value<int>(&CAMERA_WIDTH), "camera width")
		("height,t", po::value<int>(&CAMERA_HEIGHT), "camera height")
		;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help")) {
		cout << desc << endl;
		return 1;
	}
	if (!vm.count("geometry")) {
		cerr << "Missing geometry" << endl << desc << endl;;
		return 1;
	}
	if (!vm.count("camera")) {
		cerr << "Missing camera" << endl << desc << endl;;
		return 1;
	}
	if (!vm.count("width") || !vm.count("height")) {
		cerr << "Missing camera width/height" << endl << desc << endl;
		return 1;
	}
	if (vm.count("output")) {
		save = true;
	}

	cout << "geometry  : " << GeometryFileName << endl;
	cout << "camera    : " << CameraFolder << endl;
	cout << "width     : " << CAMERA_WIDTH << endl;
	cout << "height    : " << CAMERA_HEIGHT << endl;
	cout << "output    : " << OUTPUT_DIR << endl;

	try {
		/* Get a list of camera parameters */
		vector<string> exfiles, infiles, sns;
		std::vector<Intrinsic> intrinsics;
		std::vector<Extrinsic> extrinsics; 
		
		ReadExtrinsicAndIntrinsic(CameraFolder, exfiles, infiles, sns);

		if (exfiles.size() == 0) {
			throw runtime_error("No parameters found");
		}

		// Read camera parameters
		for (size_t i = 0; i < exfiles.size(); ++i) {
			intrinsics.push_back(CommonIO::ReadIntrinsic(infiles[i], CAMERA_WIDTH, CAMERA_HEIGHT));
			extrinsics.push_back(CommonIO::ReadExtrinsic(exfiles[i]));
		}

		// set up mesh and texture
		vector<Geometry> geometry = Geometry::FromObj(GeometryFileName);

		// Initialize OpenGL context
		if (InitGLContext(CAMERA_WIDTH, CAMERA_HEIGHT) < 0)
			throw runtime_error("InitGLContext fail");

		// set up renderer
		Renderer renderer;
		renderer.SetGeometries(geometry);

		// screen shot buffer
		unsigned char *buffer = new unsigned char[CAMERA_HEIGHT*CAMERA_WIDTH*3]();

		// create directories
		if (save) {
			PrepareOutputDir(OUTPUT_DIR);
		}

		// render loop
		int camIdx = 0;

		while (true) {
			if (camIdx >= extrinsics.size())
				break;

			renderer.SetCamera(Camera(extrinsics[camIdx], intrinsics[camIdx]));
			renderer.Render();

			if (save) {
				// sava screen shot
				renderer.ScreenShot(buffer, 0, 0, CAMERA_WIDTH, CAMERA_HEIGHT);
				std::ostringstream oss;
				oss.fill('0');
				oss << OUTPUT_DIR << "/" << sns[camIdx] << ".png";
				WriteImage(buffer, CAMERA_WIDTH, CAMERA_HEIGHT, oss.str());
			}

			// jump to next camera
			camIdx++; 
		}

		// release resources
		delete[] buffer;
	}
	catch (std::exception &e) {
		std::cerr << "Catch exception: " << e.what() << endl;
	}

	return 0;
}

namespace {
	int InitGLContext(int width, int height)
	{
		EGLint egl_err;
		EGLDisplay display;
		EGLint major, minor, num_config;
		EGLConfig config;
		EGLContext context;
        	EGLSurface surface;
		EGLint const attrib_list[] = {
			EGL_RED_SIZE, 8,
			EGL_GREEN_SIZE, 8,
			EGL_BLUE_SIZE, 8,
			EGL_DEPTH_SIZE, 8,
			EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
			EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
			EGL_NONE
		};
		EGLint pbuffer_attrib_list[] = {
			EGL_WIDTH, width,
			EGL_HEIGHT, height,
			EGL_NONE,
		};
		const int max_devices = 32;
		EGLDeviceEXT devices[max_devices];
		EGLint num_devices = 0;
		EGLint i;

		if ((egl_err = eglGetError()) != EGL_SUCCESS) {
			fprintf(stderr, "Bad EGL\n");
			return egl_err;
		}
		if (!gladLoadEGL()) {
			printf("gladLoadEGL fail\n");
			return -1;
		}
		// query native devices
		if (!eglQueryDevicesEXT(max_devices, devices, &num_devices) || num_devices <= 0) {
			fprintf(stderr, "eglQueryDevicesEXT fail\n");
			goto error;
		}
		for (i = 0; i < num_devices; i++) {
			// get an EGL display connection
			if ((display = eglGetPlatformDisplay(EGL_PLATFORM_DEVICE_EXT,
						 egl_devices[i], NULL)) == EGL_NO_DISPLAY)
				continue;
			// initialize the EGL display connection
			if (!eglInitialize(display, &major, &minor))
				continue;
			break;
		}
		if (i == num_devices) {
			fprintf(stderr, "eglGetDisplay returns EGL_NO_DISPLAY\n");
			goto error;
		}
		if (!eglBindAPI(EGL_OPENGL_API)) {
			fprintf(stderr, "eglBindAPI fail\n");
			goto error;
		}
		// get an appropriate EGL frame buffer configuration
        	if (!eglChooseConfig(display, attrib_list, &config, 1, &num_config)) {
			fprintf(stderr, "eglChooseConfig fail\n");
			goto error;
		}
		// create an EGL rendering context
        	if ((context = eglCreateContext(display, config, EGL_NO_CONTEXT, NULL)) == EGL_NO_CONTEXT) {
			fprintf(stderr, "eglCreateContext fail\n");
			goto error;
		}
		// create a pbuffer surface
		if ((surface = eglCreatePbufferSurface(display, config, pbuffer_attrib_list)) == EGL_NO_SURFACE) {
			fprintf(stderr, "eglCreatePbufferSurface fail\n");
			goto error;
		}
		// connect the context to the surface 
        	if (!eglMakeCurrent(display, surface, surface, context)) {
			fprintf(stderr, "eglMakeCurrent fail\n");
			goto error;
		}
		// load GL functions
		if (!gladLoadGL()) {
			fprintf(stderr, "gladLoadGLLoader fail\n");
			return -1;
		}
		// test OpenGL context is successfully created
		printf("GL Version: %d:%d\n", GLVersion.major, GLVersion.minor);
		glDeleteShader(glCreateShader(GL_VERTEX_SHADER));
		if (glGetError() != 0) {
			fprintf(stderr, "GL was not successfully setup\n");
			return -1;
		}
		return 0;

	error:
		egl_err = eglGetError();
		printf("eglGetError returns %d\n", egl_err);
		return -1;
	}

	void ReadExtrinsicAndIntrinsic(const string & root, vector<string> &extrinsics,
		vector<string> &intrinsics, vector<string> &sns)
	{
		vector<fs::path> subfolder;
		fs::path rootPath(root);

		if (!fs::is_directory(rootPath)) {
			throw std::runtime_error(root + " is not a directory");
		}

		// detect subfolders
		for (auto &p : fs::directory_iterator(rootPath)) {
			if (!fs::is_directory(p.path())) {
				continue;
			}
			subfolder.push_back(p.path());
		}

		// enumerate each subfolder and find world2depth.xml and depth_intrinsic.xml
		for (auto subf : subfolder) {
			string extrinsic = "";
			string intrinsic = "";
			string sn = "";

			for (auto file : fs::directory_iterator(subf)) {
				if (!fs::is_regular_file(file.path())) {
					continue;
				}
				else if (file.path().filename() == EXTRINSIC_PATH) {
					extrinsic = file.path().string();
				}
				else if (file.path().filename() == INTRINSIC_PATH) {
					intrinsic = file.path().string();
				}
			}

			if (!extrinsic.empty() && !intrinsic.empty()) {
				extrinsics.push_back(extrinsic);
				intrinsics.push_back(intrinsic);
				sn = subf.filename().string();
				sns.push_back(sn);
			}
		}
	}

	void PrepareOutputDir(const string &OUTPUT_DIR)
	{
		fs::remove_all(fs::path(OUTPUT_DIR));
		fs::create_directory(fs::path(OUTPUT_DIR));
	}

	void WriteImage(unsigned char *data, int width, int height, string filename)
	{
		if (data == nullptr) {
			return;
		}

		cv::Mat image(height, width, CV_16U);
		unsigned char *src = data;
		uint16_t *dst = image.ptr<uint16_t>();

		// decode color-coded depth
		for (int i = 0; i < width * height; ++i) {
			uint8_t cr = *src;
			uint8_t cg = *(src + 1);
			//uint8_t cb = *(src + 2);

			*dst = static_cast<uint16_t>((int)cr + ((int)cg << 8));
			dst++;
			src += 3;
		}

		cv::flip(image, image, 0);
		cv::imwrite(filename, image);
	}
}