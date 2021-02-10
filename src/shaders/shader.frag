#version 450

layout(location=0) in vec2 v_tex_coords;
layout(location=1) in vec3 v_position;
layout(location=2) in vec3 v_view_position;
layout(location=3) in vec3 v_light_position;

layout(location=0) out vec4 f_color;

layout(set = 0, binding = 0) uniform texture2D t_diffuse;
layout(set = 0, binding = 1) uniform sampler s_diffuse;
layout(set = 0, binding = 2) uniform texture2D t_normal;
layout(set = 0, binding = 3) uniform sampler s_normal;
layout(set = 0, binding = 4)
uniform MaterialParameter {
    vec3 u_ambient_color;
    vec3 u_diffuse_color;
    vec3 u_specular_color;
    float shininess;
};

layout(set = 2, binding = 0)
uniform Light {
    vec3 light_position;
    vec3 light_color;
};

void main() {
    vec4 object_color = texture(sampler2D(t_diffuse, s_diffuse), v_tex_coords);
    vec4 object_normal = texture(sampler2D(t_normal, s_normal), v_tex_coords);

    float ambient_strength = 0.1;
    vec3 ambient_color = light_color * ambient_strength * u_ambient_color;

    vec3 normal = normalize(object_normal.rgb * 2 - 1.0);
    vec3 light_dir = normalize(v_light_position - v_position);

    float diffuse_strength = max(dot(normal, light_dir), 0.0);
    vec3 diffuse_color = light_color * diffuse_strength * u_diffuse_color;

    vec3 view_dir = normalize(v_view_position - v_position);
    vec3 reflect_dir = reflect(-light_dir, normal);

    float specular_strength = pow(max(dot(view_dir, reflect_dir), 0.0), shininess);
    vec3 specular_color = specular_strength * light_color * u_specular_color;

    vec3 result = (ambient_color + diffuse_color + specular_color) * object_color.rgb;

    f_color = vec4(result, object_color.a);
    //f_color = vec4(1.0, 1.0, 1.0, 1.0);
}