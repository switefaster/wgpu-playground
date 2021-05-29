[[block]]
struct Transforms {
    model: mat4x4<f32>;
    model_inv_transpose: mat4x4<f32>;
};

[[group(0), binding(0)]]
var<uniform> u_transforms: Transforms;

[[block]]
struct MaterialParameter {
    ambient_color: vec3<f32>;
    diffuse_color: vec3<f32>;
    specular_color: vec3<f32>;
    shininess: f32;
};

[[group(1), binding(0)]]
var<uniform> u_material: MaterialParameter;

[[block]]
struct ViewProj {
    view_pos: vec3<f32>;
    view_proj: mat4x4<f32>;
};

[[group(2), binding(0)]]
var<uniform> u_view_proj: ViewProj;

[[block]]
struct LightViewProj {
    view_proj_tex: mat4x4<f32>;
    view_proj: mat4x4<f32>;
    light_position: vec3<f32>;
    light_color: vec3<f32>;
};

[[group(3), binding(0)]]
var<uniform> u_light_view_proj: LightViewProj;

struct VertexOutput {
    [[builtin(position)]] position: vec4<f32>;
    [[location(0)]] tex_coord: vec2<f32>;
    [[location(1)]] world_position: vec3<f32>;
    [[location(2)]] normal: vec3<f32>;
    [[location(3)]] shadow_coord: vec4<f32>;
};

[[stage(vertex)]]
fn vs_main(
    [[location(0)]] position: vec3<f32>,
    [[location(1)]] tex_coord: vec2<f32>,
    [[location(2)]] normal: vec3<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coord = tex_coord;
    let model_space: vec4<f32> = u_transforms.model * vec4<f32>(position, 1.0);
    out.world_position = model_space.xyz;
    out.normal = (u_transforms.model_inv_transpose * vec4<f32>(normal, 0.0)).xyz;
    out.shadow_coord = u_light_view_proj.view_proj_tex * model_space;
    out.position = u_view_proj.view_proj * model_space;
    return out;
}

[[group(1), binding(1)]]
var t_diffuse: texture_2d<f32>;
[[group(1), binding(2)]]
var s_diffuse: sampler;

[[group(4), binding(0)]]
var t_shadow: texture_depth_2d;
[[group(4), binding(1)]]
var s_shadow: sampler_comparison;

fn fetch_shadow(shadow_coord: vec4<f32>) -> f32 {
    let coord: vec3<f32> = shadow_coord.xyz / shadow_coord.w;
    return textureSampleCompare(t_shadow, s_shadow, coord.xy, coord.z);
}

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    let object_color: vec4<f32> = textureSample(t_diffuse, s_diffuse, in.tex_coord);
    let ambient_strength: f32 = 0.1;
    let ambient_color: vec3<f32> = u_light_view_proj.light_color * ambient_strength * u_material.ambient_color;
    let light_dir = normalize(u_light_view_proj.light_position - in.world_position);
    let diffuse_strength: f32 = max(dot(in.normal, light_dir), 0.0);
    let diffuse_color: vec3<f32> = u_light_view_proj.light_color * diffuse_strength * u_material.diffuse_color;
    let view_dir: vec3<f32> = normalize(u_view_proj.view_pos - in.world_position);
    let reflect_dir: vec3<f32> = reflect(-light_dir, in.normal);
    let specular_strength: f32 = pow(max(dot(view_dir, reflect_dir), 0.0), u_material.shininess);
    let specular_color: vec3<f32> = specular_strength * u_light_view_proj.light_color * u_material.specular_color;
    let shadow_factor: f32 = fetch_shadow(in.shadow_coord);
    let result: vec3<f32> = (ambient_color + diffuse_color + specular_color) * object_color.rgb * (shadow_factor * 0.5 + 0.5);
    return vec4<f32>(result, object_color.a);
}
