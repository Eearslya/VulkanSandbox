#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexcoord;
layout(location = 3) in vec3 inColor;

layout(binding = 0) uniform GlobalUniformBufferObject {
    mat4 view;
    mat4 proj;
    mat4 viewProj;
} ubo;

layout(binding = 1) uniform Model {
    mat4 Transform;
    vec3 CameraPos;
    vec3 Albedo;
    vec3 Metallic;
    vec3 Roughness;
    vec3 AO;
} model;

layout(location = 0) out vec3 vNormal;
layout(location = 1) out vec3 vPosition;
layout(location = 2) out vec2 vTexcoord;
layout(location = 3) out vec3 vColor;

void main() {
    gl_Position = ubo.viewProj * vec4(inPosition, 1.0);
    vNormal = mat3(transpose(inverse(model.Transform))) * inNormal;
    vPosition = vec3(model.Transform * vec4(inPosition, 1.0));
    vTexcoord = inTexcoord;
    vColor = inColor;
}