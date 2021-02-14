#version 450

layout(location=0) in vec3 a_position;
layout(location=1) in vec2 a_tex_coords;
layout(location=2) in vec3 a_normal;
layout(location=3) in vec3 a_tangent;
layout(location=4) in vec3 a_bitangent;

layout(location=0) out vec2 v_tex_coords;
layout(location=1) out vec3 v_position;
layout(location=2) out vec3 v_view_position;
layout(location=3) out vec3 v_light_position;
layout(location=4) out vec3 v_normal;
layout(location=5) out vec4 v_shadow_tex_coord;

layout(set=0, binding=0)
uniform TransformUniform {
    mat4 model_matrix;
    mat4 model_inv_transpose;
};

layout(set=2, binding=0) 
uniform ViewUniform {
    vec3 u_view_position; 
    mat4 u_view_proj;
};

layout(set=3, binding=0) 
uniform Light {
    mat4 u_view_proj_tex;
    mat4 u_light_view_proj;
    vec3 light_position;
    vec3 light_color;
};

void main() {
    v_tex_coords = a_tex_coords;
    vec4 model_space = model_matrix * vec4(a_position, 1.0);
    mat3 normal_matrix = mat3(model_inv_transpose);
    v_normal = normal_matrix * a_normal;

    v_position = model_space.xyz;
    v_view_position = u_view_position;
    v_light_position =light_position;
    v_shadow_tex_coord = u_view_proj_tex * model_space;
    gl_Position = u_view_proj * model_space;
}
 