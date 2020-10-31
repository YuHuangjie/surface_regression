#ifndef RENDERER_H
#define RENDERER_H

#include <string>

#include <glm/glm.hpp>
#include <glad/glad.h>

#include "mesh/Geometry.hpp"
#include "common/Shader.hpp"
#include "camera/Camera.hpp"

class Renderer
{
public:
	Renderer();
	virtual ~Renderer();

	void Render(int L=-1);
	void SetGeometries(const std::vector<Geometry> &);
	void SetCamera(const Camera &);
	void ScreenShot(unsigned char *buffer, unsigned int x, unsigned int y,
		unsigned int width, unsigned int height);
	
private:
	/* Clip range */
	float mNear;
	float mFar;

	/* order of Spherical Harmonics basis */
	int mMaxL;
	int mL;

	/* Drawing option */
	std::vector<size_t> mNumFaces;
	std::vector<DrawOption> mDrawOptions;

	/* shader */
	std::string mVertexCode;
	std::string mFragmentCode;
	Shader		mShader;

	/* OpenGL object */
	size_t mnMeshes;
	GLuint *mVertexArray;
	GLuint *mVertexBuffer;
	GLuint *mElementBuffer;

	/* attrib/uniform location */
	GLuint mVPLocation;
	GLuint mLLocation;
	GLuint mCamPosLocation;

	/* matrix */
	glm::mat4 mModelMatrix;
	glm::mat4 mViewMatrix;
	glm::mat4 mProjectMatrix;

	// camera
	Camera mCamera;
};

#endif