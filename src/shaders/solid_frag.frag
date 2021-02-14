#version 450

layout(location=0) in vec4 v_color;
layout(location=1) in vec3 v_position;
layout(location=2) in vec3 v_view_position;
layout(location=3) in vec3 v_light_position;
layout(location=4) in vec3 v_normal;
layout(location=5) in vec4 v_shadow_tex_coord;

layout(location=0) out vec4 f_color;

layout(set = 1, binding = 0)
uniform MaterialParameter {
    vec3 u_ambient_color;
    vec3 u_diffuse_color;
    vec3 u_specular_color;
    float shininess;
};

layout(set = 3, binding = 0)
uniform Light {
    mat4 u_view_proj_tex;
    mat4 u_light_view_proj;
    vec3 light_position;
    vec3 light_color;
};
layout(set = 4, binding = 0) uniform texture2D t_shadow;
layout(set = 4, binding = 1) uniform sampler s_shadow;

float calcShadowFactorPCF(vec4 shadow_coord) {
    shadow_coord.xyz /= shadow_coord.w;
    ivec2 dimensions = textureSize(sampler2DShadow(t_shadow, s_shadow), 0);
    float dx = 1.0f / dimensions.x;
    float percentLit = 0.0f;
    const vec3 offsets[9] = {
        vec3(-dx, -dx, 0.0), vec3(0.0f, -dx, 0.0), vec3(dx, -dx, 0.0),
        vec3(-dx, 0.0f, 0.0), vec3(0.0f, 0.0f, 0.0), vec3(dx, 0.0f, 0.0),
        vec3(-dx, dx, 0.0), vec3(0.0f, dx, 0.0), vec3(dx, dx, 0.0),
    };

    for(int i = 0; i < 9; i++) {
        percentLit += texture(sampler2DShadow(t_shadow, s_shadow), shadow_coord.xyz + offsets[i]);
    }

    return percentLit / 9.0f;
}

void main() {
    vec4 object_color = v_color;

    float ambient_strength = 0.1;
    vec3 ambient_color = light_color * ambient_strength * u_ambient_color;

    vec3 light_dir = normalize(v_light_position - v_position);

    float diffuse_strength = max(dot(v_normal, light_dir), 0.0);
    vec3 diffuse_color = light_color * diffuse_strength * u_diffuse_color;

    vec3 view_dir = normalize(v_view_position - v_position);
    vec3 reflect_dir = reflect(-light_dir, v_normal);

    float specular_strength = pow(max(dot(view_dir, reflect_dir), 0.0), shininess);
    vec3 specular_color = specular_strength * light_color * u_specular_color;

    float shadow_factor = calcShadowFactorPCF(v_shadow_tex_coord);

    vec3 result = (ambient_color + diffuse_color + specular_color) * object_color.rgb * (shadow_factor * 0.5 + 0.5);

    f_color = vec4(result, object_color.a);
    //f_color = vec4(1.0, 1.0, 1.0, 1.0);
}