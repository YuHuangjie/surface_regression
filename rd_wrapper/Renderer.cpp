#include <iostream>
#include <sstream>
#include "Renderer.h"
#include "shader/depth_vs.h"
#include "shader/depth_frag.h"

#pragma warning(disable : 4267)

Renderer::Renderer()
	: mNear(0.01f),
	mFar(10000.f),
	mnMeshes(0),
	mVertexArray(nullptr),
	mVertexBuffer(nullptr),
	mVLocation(-1),
	mVPLocation(-1),
	mModelMatrix(0.0f),
	mViewMatrix(0.0f),
	mProjectMatrix(0.0f)
{
	mVertexCode = depth_vertex_code;
	mFragmentCode = depth_fragment_code;
	mShader = Shader(mVertexCode, mFragmentCode);

	// uniform variable location
	mVLocation = glGetUniformLocation(mShader.ID, "V");
	mVPLocation = glGetUniformLocation(mShader.ID, "VP");
	mTextureLocation = glGetUniformLocation(mShader.ID, "fTexture");
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
		if (option == DrawOption::Array) {
			mNumFaces.push_back(geometries[i].GetVertices().size() / 9);
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

		// upload mesh
		glBindVertexArray(mVertexArray[i]);
		// vertex buffer 
		glBindBuffer(GL_ARRAY_BUFFER, mVertexBuffer[i]);
		glBufferData(GL_ARRAY_BUFFER, geometry.GetVertices().size() * sizeof(float),
			geometry.GetVertices().data(), GL_STATIC_DRAW);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
		// element buffer
		if (mDrawOptions[i] == DrawOption::Element) {
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mElementBuffer[i]);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, geometry.GetIndices().size() * sizeof(int),
				geometry.GetIndices().data(), GL_STATIC_DRAW);
		}

		glEnableVertexAttribArray(0);

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
	glDisable(GL_MULTISAMPLE);
	glUseProgram(mShader.ID);

	// Compute the MVP matrix from camera
	glm::mat4 VP = mProjectMatrix * mViewMatrix;
	glUniformMatrix4fv(mVPLocation, 1, GL_FALSE, &VP[0][0]);
	glUniformMatrix4fv(mVLocation, 1, GL_FALSE, &mViewMatrix[0][0]);

	// Upload default near and far
	glUniform1f(glGetUniformLocation(mShader.ID, "near"), mNear);
	glUniform1f(glGetUniformLocation(mShader.ID, "far"), mFar);

	for (size_t i = 0; i != mnMeshes; ++i) {
		glUniform1i(glGetUniformLocation(mShader.ID, "objectID"), i+1);
			
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
	glReadPixels(x, y, width, height, GL_RGBA, GL_UNSIGNED_BYTE, buffer);
}
