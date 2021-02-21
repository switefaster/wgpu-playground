#version 450

layout(location=0) in vec3 a_position;
layout(location=1) in vec4 a_color;

layout(location=0) out vec4 v_color;

layout(set = 0, binding = 0)
uniform LightViewProj {
    mat4 u_view_proj_tex;
    mat4 u_view_proj;
    vec3 u_light_position;
    vec3 u_light_color;
};

layout(set=1, binding=0)
uniform TransformUniform {
    mat4 model_matrix;
    mat4 model_inv_transpose;
};

void main() {
    v_color = a_color;
    gl_Position = u_view_proj * model_matrix * vec4(a_position, 1.0);
}
