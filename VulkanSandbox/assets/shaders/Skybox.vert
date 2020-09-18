#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 inPosition;

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    mat4 viewProj;
} ubo;

layout(location = 0) out vec3 vTexcoord;

void main() {
    gl_Position = ubo.proj * mat4(mat3(ubo.view)) * vec4(inPosition, 1.0);
    vTexcoord = inPosition;
}