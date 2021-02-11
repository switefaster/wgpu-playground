#version 450

layout(location = 0) in vec2 v_tex_coord;
layout(location = 1) in vec4 v_color;

layout(set = 0, binding = 0) uniform texture2D font_map;
layout(set = 0, binding = 1) uniform sampler font_sampler;

layout(location = 0) out vec4 f_color;

void main() {
    f_color = v_color * vec4(1.0, 1.0, 1.0, texture(sampler2D(font_map, font_sampler), v_tex_coord).r);
}
