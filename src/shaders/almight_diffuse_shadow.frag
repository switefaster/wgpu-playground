#version 450

layout(location=0) in vec2 v_tex_coord;
layout(location=0) out vec4 f_color;

layout(set = 2, binding = 1) uniform texture2D t_diffuse;
layout(set = 2, binding = 2) uniform sampler s_diffuse;

void main() {
    if(texture(sampler2D(t_diffuse, s_diffuse), v_tex_coord).a < 0.1) {
        discard;
    }
}
