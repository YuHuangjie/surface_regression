const char *sh_fragment_code =
"#version 330 core\n"

"out vec4 color;\n"
"in vec3 v_color;\n"

"void main()\n"
"{\n"
"   color = vec4(v_color, 0.f); \n"
"}\n";
