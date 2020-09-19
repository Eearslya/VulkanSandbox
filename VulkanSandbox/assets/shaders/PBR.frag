#version 450
#extension GL_ARB_separate_shader_objects : enable

#define PI 3.14159265359

layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec3 vPosition;
layout(location = 2) in vec2 vTexcoord;
layout(location = 3) in vec3 vColor;

layout(binding = 1) uniform Model {
    mat4 Transform;
    vec3 CameraPos;
    vec3 Albedo;
    float Metallic;
    float Roughness;
    float AO;
} model;

layout(binding = 4) uniform sampler2D texSampler;

layout(location = 0) out vec4 outColor;

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    float nom = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float k) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx1 = GeometrySchlickGGX(NdotV, k);
    float ggx2 = GeometrySchlickGGX(NdotL, k);

    return ggx1 * ggx2;
}

float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}

vec3 FresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

void main() {
    vec3 uLights[4] = {vec3(2.0, 2.0, 0.0), vec3(-2.0, 2.0, 0.0), vec3(2.0, -2.0, 0.0), vec3(-2.0, -2.0, 0.0)};
    vec3 uColors[4] = {vec3(1.0), vec3(1.0), vec3(1.0), vec3(1.0)};

    vec3 N = normalize(vNormal);
    vec3 V = normalize(model.CameraPos - vPosition);
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, model.Albedo, model.Metallic);

    // Direct Lighting
    vec3 Lo = vec3(0.0);
    for (int i = 0; i < 4; i++) {
        vec3 L = normalize(uLights[i] - vPosition);
        vec3 H = normalize(V + L);

        float distance = length(uLights[i] - vPosition);
        float attenuation = 1.0 / (distance * distance);
        vec3 radiance = uColors[i] * attenuation;

        float NDF = DistributionGGX(N, H, model.Roughness);
        float G = GeometrySmith(N, V, L, model.Roughness);
        vec3 F = FresnelSchlick(max(dot(H, V), 0.0), F0);

        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - model.Metallic;

        vec3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0);
        vec3 specular = numerator / max(denominator, 0.001);

        float NdotL = max(dot(N, L), 0.0);
        Lo += (kD * model.Albedo / PI + specular) * radiance * NdotL;
    }

    vec3 ambient = vec3(0.03) * model.Albedo * model.AO;
    vec3 color = ambient + Lo;

    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));

    outColor = vec4(color, 1.0);
}