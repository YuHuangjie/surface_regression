const char *depth_fragment_code =
"#version 330 core\n"

"out vec4 color;\n"
"in vec3 vertex_position;\n"

"uniform int objectID;"
"uniform mat4 V;\n"

"void main()\n"
"{\n"
// true depth (measured in model unit)
"	vec4 vertexInCameraSpace = V * vec4(vertex_position, 1.f);\n"
"	float depth = vertexInCameraSpace.z; \n"
"	int idepth = int(-depth * 1000); \n"
"	color = vec4(idepth & 0xFF, (idepth >> 8)&0xFF, (idepth >> 16)&0xFF, 255.f);\n"
"   color /= 255.f; \n"

// Render label
"	//color.r = (((objectID & 0x18) << 3) & 0xff) / 255.0;	\n"
"	//color.g = (((objectID & 0x04) << 5) & 0xff) / 255.0;	\n"
"	//color.b = (((objectID & 0x03) << 6) & 0xff) / 255.0;	\n"
"}\n";
