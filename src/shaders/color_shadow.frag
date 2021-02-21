#version 450

layout(location=0) in vec4 a_color;
layout(location=0) out vec4 f_color;

void main() {
    if(a_color.a < 0.1) {
        discard;
    }
}