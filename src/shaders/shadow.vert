#version 450

layout(location=0) in vec3 a_position;

layout(set = 0, binding = 0)
uniform LightViewProj {
    mat4 u_view_proj_tex;
    mat4 u_view_proj;
    vec3 u_light_position;
    vec3 u_light_color;
};

void main() {
    gl_Position = u_view_proj * vec4(a_position, 1.0);
}
