[[block]]
struct LightViewProj {
    view_proj_tex: mat4x4<f32>;
    view_proj: mat4x4<f32>;
    light_position: vec3<f32>;
    light_color: vec3<f32>;
};

[[group(0), binding(0)]]
var<uniform> u_light_view_proj: LightViewProj;

[[block]]
struct Transforms {
    model: mat4x4<f32>;
    model_inv_transpose: mat4x4<f32>;
};

[[group(1), binding(0)]]
var<uniform> u_transforms: Transforms;

struct VertexOutput {
    [[builtin(position)]] position: vec4<f32>;
    [[location(0)]] tex_coord: vec2<f32>;
};

[[stage(vertex)]]
fn vs_main(
    [[location(0)]] position: vec3<f32>,
    [[location(1)]] tex_coord: vec2<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coord = tex_coord;
    out.position = u_light_view_proj.view_proj * u_transforms.model * vec4<f32>(position, 1.0);
    return out;
}

[[group(2), binding(1)]]
var t_diffuse: texture_2d<f32>;
[[group(2), binding(2)]]
var s_diffuse: sampler;

[[stage(fragment)]]
fn fs_main(in: VertexOutput) {
    if (textureSample(t_diffuse, s_diffuse, in.tex_coord).a < 0.1) {
        discard;
    }
}
