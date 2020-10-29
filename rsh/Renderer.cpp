#include <iostream>
#include <sstream>
#include "Renderer.h"
#include "shader/sh_vs.h"
#include "shader/sh_frag.h"

#pragma warning(disable : 4267)

Renderer::Renderer()
	: mNear(0.1f),
	mFar(10.f),	// change accordingly
	mL(0),
	mnMeshes(0),
	mVertexArray(nullptr),
	mVertexBuffer(nullptr),
	mVPLocation(-1),
	mLLocation(-1),
	mCamPosLocation(-1),
	mModelMatrix(0.0f),
	mViewMatrix(0.0f),
	mProjectMatrix(0.0f)
{
	mVertexCode = sh_vertex_code;
	mFragmentCode = sh_fragment_code;
	mShader = Shader(mVertexCode, mFragmentCode);

	// uniform variable location
	mVPLocation = glGetUniformLocation(mShader.ID, "VP");
	mLLocation = glGetUniformLocation(mShader.ID, "L");
	mCamPosLocation = glGetUniformLocation(mShader.ID, "cam_pos");
}

Renderer::~Renderer()
{
	glDeleteProgram(mShader.ID);
	glDeleteVertexArrays(mnMeshes, mVertexArray);
	glDeleteBuffers(mnMeshes, mVertexBuffer);
	glDeleteBuffers(mnMeshes, mElementBuffer);
}

void Renderer::SetGeometries(const std::vector<Geometry> &geometries)
{
	if (mVertexArray) {
		glDeleteVertexArrays(mnMeshes, mVertexArray);
		mVertexArray = nullptr;
	}
	if (mVertexBuffer) {
		glDeleteBuffers(mnMeshes, mVertexBuffer);
		mVertexBuffer = nullptr;
	}
	if (mElementBuffer) {
		glDeleteBuffers(mnMeshes, mElementBuffer);
		mElementBuffer = nullptr;
	}
	mVertexArray = new GLuint[geometries.size()];
	mVertexBuffer = new GLuint[geometries.size()];
	mElementBuffer = new GLuint[geometries.size()];

	glGenVertexArrays(static_cast<GLsizei>(geometries.size()), mVertexArray);
	glGenBuffers(static_cast<GLsizei>(geometries.size()), mVertexBuffer);
	glGenBuffers(static_cast<GLsizei>(geometries.size()), mElementBuffer);

	mNumFaces.clear();
	mDrawOptions.clear();
	for (int i = 0; i != geometries.size(); ++i) {
		DrawOption option = geometries[i].GetDrawOption();
		int nva = geometries[i].nVertexAttribs;

		if (nva == 6) mL = 0;
		else if (nva == 15) mL = 1;
		else if (nva == 30) mL = 2;
		else if (nva == 51) mL = 3;
		else throw runtime_error("wrong number of vertex attributes");

		if (option == DrawOption::Array) {
			mNumFaces.push_back(geometries[i].GetVertices().size() / nva);
		}
		else {
			mNumFaces.push_back(geometries[i].GetIndices().size() / 3);
		}
		mDrawOptions.push_back(option);
		
		// Output geometry information
		std::ostringstream oss;
		oss << "Read #" << i << " mesh: " << mNumFaces[i] << " faces. "
			<< "DrawOption: ";
		if (option == DrawOption::Array) {
			oss << "Array";
		}
		else if (option == DrawOption::Element) {
			oss << "Element";
		}

		std::cout << oss.str() << std::endl;
	}
	
	for (int i = 0; i != geometries.size(); ++i) {
		const Geometry &geometry = geometries[i];
		int nva = geometry.nVertexAttribs;

		// upload mesh
		glBindVertexArray(mVertexArray[i]);
		// vertex buffer 
		glBindBuffer(GL_ARRAY_BUFFER, mVertexBuffer[i]);
		glBufferData(GL_ARRAY_BUFFER, geometry.GetVertices().size() * sizeof(float),
			geometry.GetVertices().data(), GL_STATIC_DRAW);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, nva*sizeof(float), (void*)0);
		// coefficients of order 0 SH
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, nva*sizeof(float), (void*)(3*sizeof(float)));

		if (mL >= 1) {
			int vab = 2;
			int vae = (mL == 1 ? 4 : 9);
			for (int j = vab; j <= vae; j++) {
				glVertexAttribPointer(j, 3, GL_FLOAT, GL_FALSE, nva*sizeof(float), (void*)(3*j*sizeof(float)));
				glEnableVertexAttribArray(j);
			}
		}
		if (mL == 3) {
			int vab = 10;
			int vae = 15;
			int p = 30;
			for (int j = vab; j <= vae; j++) {
				glVertexAttribPointer(j, 4, GL_FLOAT, GL_FALSE, nva*sizeof(float), (void*)((p+(j-vab)*4)*sizeof(float)));
				glEnableVertexAttribArray(j);
			}
		}

		// element buffer
		if (mDrawOptions[i] == DrawOption::Element) {
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mElementBuffer[i]);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, geometry.GetIndices().size() * sizeof(int),
				geometry.GetIndices().data(), GL_STATIC_DRAW);
		}

		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}

	mnMeshes = geometries.size();
}

void Renderer::Render(void)
{
	glCullFace(GL_BACK);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);  
	glCullFace(GL_BACK);
	glDisable(GL_MULTISAMPLE);
	glUseProgram(mShader.ID);

	// Compute the MVP matrix from camera
	glm::mat4 VP = mProjectMatrix * mViewMatrix;
	glm::vec3 cam_pos = mCamera.GetPosition();
	glUniformMatrix4fv(mVPLocation, 1, GL_FALSE, &VP[0][0]);
	glUniform3f(mCamPosLocation, cam_pos.x, cam_pos.y, cam_pos.z);
	glUniform1i(mLLocation, mL);

	for (size_t i = 0; i != mnMeshes; ++i) {
		// 1rst attribute buffer: position
		glBindVertexArray(mVertexArray[i]);

		// Draw the triangles
		if (mDrawOptions[i] == DrawOption::Array) {
			glDrawArrays(GL_TRIANGLES, 0, mNumFaces[i] * 3);
		}
		else if (mDrawOptions[i] == DrawOption::Element) {
			glDrawElements(GL_TRIANGLES, mNumFaces[i] * 3, GL_UNSIGNED_INT, 0);
		}

		glBindVertexArray(0);
	}
}

void Renderer::SetCamera(const Camera &camera)
{
	mCamera = camera;
	mModelMatrix = glm::mat4(1.0f);
	mViewMatrix = mCamera.GetViewMatrix();
	mProjectMatrix = mCamera.GetProjectionMatrix(mNear, mFar);
}

void Renderer::ScreenShot(unsigned char *buffer, unsigned int x, unsigned int y,
	unsigned int width, unsigned int height)
{
	if (buffer == nullptr) {
		return;
	}
	glReadPixels(x, y, width, height, GL_RGB, GL_UNSIGNED_BYTE, buffer);
}
