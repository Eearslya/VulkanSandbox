// ======================================================================
// Preprocessors, Macros, Includes
// ======================================================================

// Defines / Constants
#define U32_MAX 0xfffffffful
#define U64_MAX 0xffffffffffffffffull
#define NOMINMAX
#define TINYOBJLOADER_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define WIN32_LEAN_AND_MEAN
#define FORCEINLINE __forceinline
#define FORCENOINLINE _declspec(noinline)
#define ALIGN(n) __declspec(align(n))
#define TRACE
#if defined(_DEBUG) || defined(TRACE)
#define DEBUG
#endif
#ifndef NO_ASSERT
#define ASSERT_ENABLE
#endif

// Includes
#include <Windows.h>
#include <stb_image.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <tiny_obj_loader.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_win32.h>

#include <chrono>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>
#include <iostream>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

// Macros
#ifdef ASSERT_ENABLE
#if _MSC_VER
#include <intrin.h>
#define DoDebugBreak() __debugbreak();
#else
#define DoDebugBreak() __asm {int 3}
#endif

#define ASSERT_MSG(expr, msg)                           \
  {                                                     \
    if (expr) {                                         \
    } else {                                            \
      AssertionFailure(#expr, msg, __FILE__, __LINE__); \
      DoDebugBreak();                                   \
    }                                                   \
  }

#define ASSERT(expr) ASSERT_MSG(expr, "")

#ifdef DEBUG
#define ASSERT_DEBUG(expr) ASSERT(expr)
#define ASSERT_DEBUG_MSG(expr, msg) ASSERT_MSG(expr, msg)
#else
#define ASSERT_DEBUG(expr)
#define ASSERT_DEBUG_MSG(expr, msg)
#endif

FORCEINLINE void AssertionFailure(const char* expression, const char* msg, const char* file,
                                  int line) {
  std::cerr << "Assertion Failure: " << expression << "\n";
  std::cerr << "  Message: " << msg << "\n";
  std::cerr << "  At: " << file << ":" << line << "\n";
}

#else
#define ASSERT(expr)
#define ASSERT_MSG(expr, msg)
#define ASSERT_DEBUG(expr)
#define ASSERT_DEBUG_MSG(expr, msg)
#endif

#define VkCall(expr)                                            \
  {                                                             \
    if (expr != VK_SUCCESS) {                                   \
      throw std::runtime_error("!!! VULKAN EXCEPTION: " #expr); \
    }                                                           \
  }

#define ArrLen(arr) sizeof(arr) / sizeof(*arr)

// Typedefs
typedef uint64_t U64;
typedef uint32_t U32;
typedef uint16_t U16;
typedef uint8_t U8;

typedef int64_t I64;
typedef int32_t I32;
typedef int16_t I16;
typedef int8_t I8;

typedef double F64;
typedef float F32;

// ======================================================================
// Logging
// ======================================================================
static const char* LogLevelNames[] = {"FATAL", "ERROR", "WARN", "INFO", "DEBUG", "TRACE"};
static char LogBuffer[8192];
enum class LogLevel { Fatal = 0, Error, Warn, Info, Debug, Trace };

#ifdef TRACE
LogLevel _logLevel = LogLevel::Trace;
#elif defined(DEBUG)
LogLevel _logLevel = LogLevel::Debug;
#else
LogLevel _logLevel = LogLevel::Info;
#endif

void WriteLog(LogLevel level, const char* message, ...) {
  if (level > _logLevel) {
    return;
  }
  va_list args;
  va_start(args, message);
  vsnprintf(LogBuffer, sizeof(LogBuffer), message, args);
  fprintf(stdout, "[%s]: %s\r\n", LogLevelNames[static_cast<U32>(level)], LogBuffer);
  fflush(stdout);
  va_end(args);
}

#define LogFatal(msg, ...) WriteLog(LogLevel::Fatal, msg, __VA_ARGS__)
#define LogError(msg, ...) WriteLog(LogLevel::Error, msg, __VA_ARGS__)
#define LogWarn(msg, ...) WriteLog(LogLevel::Warn, msg, __VA_ARGS__)
#define LogInfo(msg, ...) WriteLog(LogLevel::Info, msg, __VA_ARGS__)
#define LogDebug(msg, ...) WriteLog(LogLevel::Debug, msg, __VA_ARGS__)
#define LogTrace(msg, ...) WriteLog(LogLevel::Trace, msg, __VA_ARGS__)

// ======================================================================
// Data Structures
// ======================================================================
struct VulkanPhysicalDeviceSwapchainSupport {
  VkSurfaceCapabilitiesKHR Capabilities;
  std::vector<VkSurfaceFormatKHR> Formats;
  std::vector<VkPresentModeKHR> PresentationModes;
};

struct VulkanPhysicalDeviceQueue {
  U32 Index;
  VkQueueFlags Flags;
  U32 Count;
  VkBool32 PresentKHR;

  const bool SupportsGraphics() const { return Flags & VK_QUEUE_GRAPHICS_BIT; }
  const bool SupportsCompute() const { return Flags & VK_QUEUE_COMPUTE_BIT; }
  const bool SupportsTransfer() const { return Flags & VK_QUEUE_TRANSFER_BIT; }
  const bool SupportsSparseBinding() const { return Flags & VK_QUEUE_SPARSE_BINDING_BIT; }
  const bool SupportsProtected() const { return Flags & VK_QUEUE_PROTECTED_BIT; }
  const bool SupportsPresentation() const { return PresentKHR; }
};

struct VulkanPhysicalDeviceQueues {
  std::vector<VulkanPhysicalDeviceQueue> Queues;

  U32 Count;
  I32 GraphicsIndex;
  I32 TransferIndex;
  I32 ComputeIndex;
  I32 PresentationIndex;
};

struct VulkanPhysicalDeviceInfo {
  VkPhysicalDevice Device;
  VkPhysicalDeviceFeatures Features;
  VkPhysicalDeviceMemoryProperties Memory;
  VkPhysicalDeviceProperties Properties;
  VulkanPhysicalDeviceQueues Queues;
  VulkanPhysicalDeviceSwapchainSupport SwapchainSupport;
  std::vector<VkExtensionProperties> Extensions;
};

struct Vertex {
  glm::vec3 Position;
  glm::vec3 Normal;
  glm::vec2 TexCoord;
  glm::vec3 Color;

  bool operator==(const Vertex& other) const {
    return Position == other.Position && Normal == other.Normal && TexCoord == other.TexCoord &&
           Color == other.Color;
  }
};

namespace std {
template <>
struct hash<Vertex> {
  size_t operator()(Vertex const& vertex) const {
    glm::mat4 mat;
    mat[0] = {vertex.Position.x, vertex.Position.y, vertex.Position.z, vertex.TexCoord.x};
    mat[1] = {vertex.Normal.x, vertex.Normal.y, vertex.Normal.z, vertex.TexCoord.y};
    mat[2] = {vertex.Color.x, vertex.Color.y, vertex.Color.z, 0.0f};
    return hash<glm::mat4>{}(mat);
  }
};
}  // namespace std

struct UniformBufferObject {
  glm::mat4 Model;
  glm::mat4 View;
  glm::mat4 Projection;
};

struct Mesh {
  std::vector<Vertex> Vertices;
  std::vector<U32> Indices;
  VkBuffer Buffer = VK_NULL_HANDLE;
  VkDeviceMemory DeviceMemory = VK_NULL_HANDLE;
  VkDeviceSize IndicesOffset = 0;

  Mesh(const std::vector<Vertex>& vertices, const std::vector<U32>& indices) {
    Vertices = vertices;
    Indices = indices;
  }

  Mesh(const std::string& objFile) {
    tinyobj::attrib_t objectAttribs;
    std::vector<tinyobj::shape_t> objectShapes;
    std::vector<tinyobj::material_t> objectMaterials;
    std::string objectWarning;
    std::string objectError;

    LogDebug("Loading object from file: %s", objFile.c_str());

    bool objectLoaded =
        tinyobj::LoadObj(&objectAttribs, &objectShapes, &objectMaterials, &objectWarning,
                         &objectError, objFile.c_str(), "assets/models");
    if (!objectWarning.empty()) {
      LogWarn("TinyObjLoader: %s", objectWarning.c_str());
    }
    if (!objectError.empty()) {
      LogError("TinyObjLoader: %s", objectError.c_str());
    }

    LogDebug("Loaded object \"%s\"", objFile.c_str());
    LogDebug(" - Vertex Count:   %d", objectAttribs.vertices.size() / 3);
    LogDebug(" - Normal Count:   %d", objectAttribs.normals.size() / 3);
    LogDebug(" - UV Count:       %d", objectAttribs.texcoords.size() / 2);
    LogDebug(" - Material Count: %d", objectMaterials.size());

    std::unordered_map<Vertex, U32> uniqueVertices{};
    for (const auto& shape : objectShapes) {
      for (const auto& index : shape.mesh.indices) {
        Vertex vertex{};

        vertex.Position = {objectAttribs.vertices[3 * index.vertex_index + 0],
                           objectAttribs.vertices[3 * index.vertex_index + 1],
                           objectAttribs.vertices[3 * index.vertex_index + 2]};
        vertex.Normal = {objectAttribs.normals[3 * index.normal_index + 0],
                         objectAttribs.normals[3 * index.normal_index + 1],
                         objectAttribs.normals[3 * index.normal_index + 2]};
        if (index.texcoord_index != -1) {
          vertex.TexCoord = {objectAttribs.texcoords[2 * index.texcoord_index + 0],
                             1.0f - objectAttribs.texcoords[2 * index.texcoord_index + 1]};
        }
        vertex.Color = {1.0f, 1.0f, 1.0f};

        if (uniqueVertices.count(vertex) == 0) {
          uniqueVertices[vertex] = static_cast<U32>(Vertices.size());
          Vertices.push_back(vertex);
        }

        Indices.push_back(uniqueVertices[vertex]);
      }
    }
  }
};

struct Material;
struct Object;

// ======================================================================
// Global State Structures
// ======================================================================
constexpr const static struct Configuration {
  // Constant / configuration values
  U32 WindowWidth = 1600;
  U32 WindowHeight = 900;
  const wchar_t* WindowTitle = L"Vulkan";
  const wchar_t* WindowClassName = L"VulkanWindow";
  U32 MaxImagesInFlight = 2;
} config;

static struct ApplicationContext {
  HINSTANCE Instance = NULL;
  HWND Window = NULL;
  HCURSOR Cursor = NULL;
  bool CloseRequested = false;
} app;

static struct VulkanContext {
  VkInstance Instance = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT DebugMessenger = VK_NULL_HANDLE;
  VkPhysicalDevice PhysicalDevice = VK_NULL_HANDLE;
  VkDevice Device = VK_NULL_HANDLE;
  VulkanPhysicalDeviceInfo PhysicalDeviceInfo{};
  VkQueue GraphicsQueue = VK_NULL_HANDLE;
  VkQueue TransferQueue = VK_NULL_HANDLE;
  VkQueue PresentationQueue = VK_NULL_HANDLE;
  VkCommandPool GraphicsCommandPool = VK_NULL_HANDLE;
  VkCommandPool TransferCommandPool = VK_NULL_HANDLE;
  VkSurfaceKHR Surface = VK_NULL_HANDLE;
  VkSurfaceFormatKHR SurfaceFormat{};
  VkPresentModeKHR PresentMode = VK_PRESENT_MODE_FIFO_KHR;
  VkExtent2D SwapchainExtent{};
  U32 SwapchainImageCount = 2;
  VkSwapchainKHR Swapchain = VK_NULL_HANDLE;
  std::vector<VkImage> SwapchainImages;
  std::vector<VkImageView> SwapchainImageViews;
  VkRenderPass RenderPass = VK_NULL_HANDLE;
  VkViewport Viewport{};
  VkRect2D Scissor{};
  std::vector<VkFramebuffer> Framebuffers;
  std::vector<VkCommandBuffer> CommandBuffers;
  std::vector<VkSemaphore> ImageAvailableSemaphores;
  std::vector<VkSemaphore> RenderFinishedSemaphores;
  std::vector<VkFence> InFlightFences;
  std::vector<VkFence> ImagesInFlight;
  VkFormat DepthFormat;
  VkImage DepthImage = VK_NULL_HANDLE;
  VkDeviceMemory DepthImageMemory = VK_NULL_HANDLE;
  VkImageView DepthImageView = VK_NULL_HANDLE;
  std::list<Object*> ObjectsToRender;
} vk;

// ======================================================================
// Function Declarations
// ======================================================================
static VKAPI_ATTR VkBool32 VKAPI_CALL
VulkanDebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                    VkDebugUtilsMessageTypeFlagsEXT messageType,
                    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData);
LRESULT CALLBACK ApplicationWindowProcedure(HWND hwnd, U32 msg, WPARAM wParam, LPARAM lParam);
VkShaderModule CreateShaderModule(VkDevice vkDevice, const std::string& sourceFile);
bool CreateImage(U32 width, U32 height, VkFormat format, VkImageTiling tiling,
                 VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image,
                 VkDeviceMemory& memory);
void TransitionImageLayout(VkImage image, VkFormat format, VkImageLayout srcLayout,
                           VkImageLayout dstLayout);
void CreateSwapchain();
inline U32 FindMemoryType(U32 typeFilter, VkMemoryPropertyFlags properties);
inline VkFormat FindImageFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling,
                                VkFormatFeatureFlags features);
bool CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                  VkBuffer& buffer, VkDeviceMemory& deviceMemory, U64* bufferAlignment = nullptr);
void CopyBuffer(VkBuffer source, VkBuffer destination, VkDeviceSize size);
void CopyBufferToImage(VkBuffer source, VkImage destination, U32 width, U32 height);
bool UploadMesh(Mesh& mesh);
void FreeMesh(const Mesh& mesh);
void Run();

// ======================================================================
// Materials
// ======================================================================

struct Material {
  virtual ~Material() = default;
  virtual void StaticInitialize() = 0;
  virtual void StaticDestroy() = 0;
  virtual void Bind(VkCommandBuffer& cmdBuf, U32 imageIndex) = 0;
  virtual void Update(U32 imageIndex) = 0;
};

struct UnlitMaterial : public Material {
  static bool Initialized;
  static U32 RefCount;
  static VkDescriptorSetLayout DescriptorSetLayout;
  static VkPipelineLayout PipelineLayout;
  static VkPipeline Pipeline;

  VkDescriptorPool DescriptorPool = VK_NULL_HANDLE;
  std::vector<VkBuffer> UniformBuffers;
  std::vector<VkDeviceMemory> UniformBufferMemories;
  std::vector<VkDescriptorSet> DescriptorSets;

  UnlitMaterial() {
    if (!Initialized) {
      StaticInitialize();
    }

    // DescriptorPool
    {
      const VkDescriptorPoolSize poolSizes[] = {{
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,  // type
          vk.SwapchainImageCount              // descriptorCount
      }};

      const VkDescriptorPoolCreateInfo createInfo{
          VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,  // sType
          nullptr,                                        // pNext
          0,                                              // flags
          vk.SwapchainImageCount,                         // maxSets
          ArrLen(poolSizes),                              // poolSizeCount
          poolSizes                                       // pPoolSizes
      };
      VkCall(vkCreateDescriptorPool(vk.Device, &createInfo, nullptr, &DescriptorPool));
    }

    // Uniform Buffers
    {
      VkDeviceSize bufferSize = sizeof(UniformBufferObject);
      UniformBuffers.resize(vk.SwapchainImageCount, VK_NULL_HANDLE);
      UniformBufferMemories.resize(vk.SwapchainImageCount, VK_NULL_HANDLE);

      for (U32 i = 0; i < vk.SwapchainImageCount; i++) {
        CreateBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     UniformBuffers[i], UniformBufferMemories[i]);
      }
    }

    // Descriptor Sets
    {
      std::vector<VkDescriptorSetLayout> layouts(vk.SwapchainImageCount, DescriptorSetLayout);

      const VkDescriptorSetAllocateInfo allocateInfo{
          VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,  // sType
          nullptr,                                         // pNext
          DescriptorPool,                                  // decriptorPool
          vk.SwapchainImageCount,                          // descriptorSetCount
          layouts.data()                                   // pSetLayouts
      };

      DescriptorSets.resize(vk.SwapchainImageCount, VK_NULL_HANDLE);
      VkCall(vkAllocateDescriptorSets(vk.Device, &allocateInfo, DescriptorSets.data()));

      for (U32 i = 0; i < vk.SwapchainImageCount; i++) {
        const VkDescriptorBufferInfo bufferInfo{
            UniformBuffers[i],           // buffer
            0,                           // offset
            sizeof(UniformBufferObject)  // range
        };

        const VkWriteDescriptorSet descriptorWrites[] = {{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,  // sType
            nullptr,                                 // pNext
            DescriptorSets[i],                       // dstSet
            0,                                       // dstBinding
            0,                                       // dstArrayElement
            1,                                       // descriptorCount
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,       // descriptorType
            nullptr,                                 // pImageInfo
            &bufferInfo,                             // pBufferInfo
            nullptr                                  // pTexelBufferView

        }};

        vkUpdateDescriptorSets(vk.Device, ArrLen(descriptorWrites), descriptorWrites, 0, nullptr);
      }
    }
  }

  ~UnlitMaterial() {
    for (U32 i = 0; i < vk.SwapchainImageCount; i++) {
      vkDestroyBuffer(vk.Device, UniformBuffers[i], nullptr);
      vkFreeMemory(vk.Device, UniformBufferMemories[i], nullptr);
    }
    vkDestroyDescriptorPool(vk.Device, DescriptorPool, nullptr);

    if (--RefCount == 0) {
      StaticDestroy();
    }
  }

  void StaticInitialize() override {
    // DescriptorSetLayout
    {
      constexpr static const VkDescriptorSetLayoutBinding uboLayoutBinding{
          0,                                  // binding
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,  // descriptorType
          1,                                  // descriptorCount
          VK_SHADER_STAGE_VERTEX_BIT,         // stageFlags
          nullptr                             // pImmutableSamplers
      };

      constexpr static const VkDescriptorSetLayoutBinding bindings[] = {uboLayoutBinding};

      constexpr static const VkDescriptorSetLayoutCreateInfo layoutCreateInfo{
          VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,  // sType
          nullptr,                                              // pNext
          0,                                                    // flags
          sizeof(bindings) / sizeof(*bindings),                 // bindingCount
          bindings                                              // pBindings
      };
      VkCall(
          vkCreateDescriptorSetLayout(vk.Device, &layoutCreateInfo, nullptr, &DescriptorSetLayout));
    }

    // Pipeline
    {
      VkShaderModule vertexShaderModule =
          CreateShaderModule(vk.Device, "assets/shaders/Unlit.vert.spv");
      VkShaderModule fragmentShaderModule =
          CreateShaderModule(vk.Device, "assets/shaders/Unlit.frag.spv");

      constexpr static const VkVertexInputBindingDescription vertexBindingDescription{
          0,                           // binding
          sizeof(Vertex),              // stride
          VK_VERTEX_INPUT_RATE_VERTEX  // inputRate
      };

      constexpr static const VkVertexInputAttributeDescription vertexAttributeDescription[]{
          {
              0,                           // location
              0,                           // binding
              VK_FORMAT_R32G32B32_SFLOAT,  // format
              offsetof(Vertex, Position)   // offset
          },                               // Position
          {
              1,                           // location
              0,                           // binding
              VK_FORMAT_R32G32B32_SFLOAT,  // format
              offsetof(Vertex, Normal)     // offset
          },                               // Normal
          {
              2,                          // location
              0,                          // binding
              VK_FORMAT_R32G32_SFLOAT,    // format
              offsetof(Vertex, TexCoord)  // offset
          },                              // TexCoord
          {
              3,                           // location
              0,                           // binding
              VK_FORMAT_R32G32B32_SFLOAT,  // format
              offsetof(Vertex, Color)      // offset
          },                               // Color
      };

      const VkPipelineShaderStageCreateInfo vertexShaderStageCreateInfo{
          VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,  // sType
          nullptr,                                              // pNext
          0,                                                    // flags
          VK_SHADER_STAGE_VERTEX_BIT,                           // stage
          vertexShaderModule,                                   // module
          "main",                                               // pName
          nullptr                                               // pSpecializationInfo
      };

      const VkPipelineShaderStageCreateInfo fragmentShaderStageCreateInfo{
          VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,  // sType
          nullptr,                                              // pNext
          0,                                                    // flags
          VK_SHADER_STAGE_FRAGMENT_BIT,                         // stage
          fragmentShaderModule,                                 // module
          "main",                                               // pName
          nullptr                                               // pSpecializationInfo
      };

      const VkPipelineShaderStageCreateInfo shaderStageCreateInfos[] = {
          vertexShaderStageCreateInfo, fragmentShaderStageCreateInfo};

      constexpr static const VkPipelineVertexInputStateCreateInfo vertexInputInfo{
          VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,  // sType
          nullptr,                                                    // pNext
          0,                                                          // flags
          1,                          // vertexBindingDescriptionCount
          &vertexBindingDescription,  // pVertexBindingDescriptions
          sizeof(vertexAttributeDescription) /
              sizeof(*vertexAttributeDescription),  // vertexAttributeDescriptionCount
          vertexAttributeDescription                // pVertexAttributeDescriptions
      };

      constexpr static const VkPipelineInputAssemblyStateCreateInfo inputAssembly{
          VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,  // sType
          nullptr,                                                      // pNext
          0,                                                            // flags
          VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,                          // topology
          VK_FALSE                                                      // primitiveRestartEnable
      };

      const VkPipelineViewportStateCreateInfo viewportState{
          VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,  // sType
          nullptr,                                                // pNext
          0,                                                      // flags
          1,                                                      // viewportCount
          nullptr,                                                // pViewports
          1,                                                      // scissorCount
          nullptr                                                 // pScissors
      };

      constexpr static const VkPipelineRasterizationStateCreateInfo rasterizer{
          VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,  // sType
          nullptr,                                                     // pNext
          0,                                                           // flags
          VK_FALSE,                                                    // depthClampEnable
          VK_FALSE,                                                    // rasterizerDiscardEnable
          VK_POLYGON_MODE_FILL,                                        // polygonMode
          VK_CULL_MODE_BACK_BIT,                                       // cullMode
          VK_FRONT_FACE_COUNTER_CLOCKWISE,                             // frontFace
          VK_FALSE,                                                    // depthBiasEnable
          0.0f,                                                        // depthBiasConstantFactor
          0.0f,                                                        // depthBiasClamp
          0.0f,                                                        // depthBiasSlopeFactor
          1.0f                                                         // lineWidth
      };

      constexpr static const VkPipelineMultisampleStateCreateInfo multisampling{
          VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,  // sType
          nullptr,                                                   // pNext
          0,                                                         // flags
          VK_SAMPLE_COUNT_1_BIT,                                     // rasterizationSamples
          VK_FALSE,                                                  // sampleShadingEnable
          1.0f,                                                      // minSampleShading
          nullptr,                                                   // pSampleMask
          VK_FALSE,                                                  // alphaToCoverageEnable
          VK_FALSE                                                   // alphaToOneEnable
      };

      constexpr static const VkPipelineColorBlendAttachmentState colorBlendAttachment{
          VK_TRUE,                              // blendEnable
          VK_BLEND_FACTOR_SRC_ALPHA,            // srcColorBlendFactor
          VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,  // dstColorBlendFactor
          VK_BLEND_OP_ADD,                      // colorBlendOp
          VK_BLEND_FACTOR_ONE,                  // srcAlphaBlendFactor
          VK_BLEND_FACTOR_ZERO,                 // dstAlphaBlendFactor
          VK_BLEND_OP_ADD,                      // alphaBlendOp
          VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
              VK_COLOR_COMPONENT_A_BIT  // colorWriteMask
      };

      constexpr static const VkPipelineColorBlendStateCreateInfo colorBlending{
          VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,  // sType
          nullptr,                                                   // pNext
          0,                                                         // flags
          VK_FALSE,                                                  // logicOpEnable
          VK_LOGIC_OP_COPY,                                          // logicOp
          1,                                                         // attachmentCount
          &colorBlendAttachment,                                     // pAttachment
          {0.0f, 0.0f, 0.0f, 0.0f}                                   // blendConstants
      };

      constexpr static const VkPipelineDepthStencilStateCreateInfo depthStencil{
          VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,  // sType
          nullptr,                                                     // pNext
          0,                                                           // flags
          VK_TRUE,                                                     // depthTestEnable
          VK_TRUE,                                                     // depthWriteEnable
          VK_COMPARE_OP_LESS,                                          // depthCompareOp
          VK_FALSE,                                                    // depthBoundsTestEnable
          VK_FALSE,                                                    // stencilTestEnable
          {},                                                          // front
          {},                                                          // back
          0.0f,                                                        // minDepthBounds
          1.0f                                                         // maxDepthBounds
      };

      std::vector<VkDynamicState> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT,
                                                   VK_DYNAMIC_STATE_SCISSOR};

      const VkPipelineDynamicStateCreateInfo dynamicState{
          VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,  // sType
          nullptr,                                               // pNext
          0,                                                     // flags
          static_cast<U32>(dynamicStates.size()),                // dynamicStateCount
          dynamicStates.data()                                   // pDynamicStates
      };

      const VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{
          VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,  // sType
          nullptr,                                        // pNext
          0,                                              // flags
          1,                                              // setLayoutCount
          &DescriptorSetLayout,                           // pSetLayouts
          0,                                              // pushConstantRangeCount
          nullptr                                         // pPushConstantRanges
      };

      VkCall(
          vkCreatePipelineLayout(vk.Device, &pipelineLayoutCreateInfo, nullptr, &PipelineLayout));

      const VkGraphicsPipelineCreateInfo pipelineCreateInfo{
          VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,  // sType
          nullptr,                                          // pNext
          0,                                                // flags
          2,                                                // stageCount
          shaderStageCreateInfos,                           // pStages
          &vertexInputInfo,                                 // pVertexInputState
          &inputAssembly,                                   // pInputAssemblyState
          nullptr,                                          // pTessellationState
          &viewportState,                                   // pViewportState
          &rasterizer,                                      // pRasterizationState
          &multisampling,                                   // pMultisampleState
          &depthStencil,                                    // pDepthStencilState
          &colorBlending,                                   // pColorBlendState
          &dynamicState,                                    // pDynamicState
          PipelineLayout,                                   // layout
          vk.RenderPass,                                    // renderPass
          0,                                                // subpass,
          VK_NULL_HANDLE,                                   // basePipelineHandle
          -1                                                // basePipelineIndex
      };

      VkCall(vkCreateGraphicsPipelines(vk.Device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr,
                                       &Pipeline));

      vkDestroyShaderModule(vk.Device, fragmentShaderModule, nullptr);
      vkDestroyShaderModule(vk.Device, vertexShaderModule, nullptr);
    }

    RefCount = 1;
    Initialized = true;
  }

  void StaticDestroy() override {
    vkDestroyPipeline(vk.Device, Pipeline, nullptr);
    vkDestroyPipelineLayout(vk.Device, PipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(vk.Device, DescriptorSetLayout, nullptr);

    Pipeline = VK_NULL_HANDLE;
    PipelineLayout = VK_NULL_HANDLE;
    DescriptorSetLayout = VK_NULL_HANDLE;
    Initialized = false;
    RefCount = 0;
  }

  void Bind(VkCommandBuffer& cmdBuf, U32 imageIndex) override {
    // vkCmdBindPipeline
    { vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, Pipeline); }

    // Update viewport/scissor
    vkCmdSetViewport(cmdBuf, 0, 1, &vk.Viewport);
    vkCmdSetScissor(cmdBuf, 0, 1, &vk.Scissor);

    // Bind descriptor set
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, PipelineLayout, 0, 1,
                            &DescriptorSets[imageIndex], 0, nullptr);
  }

  void Update(U32 imageIndex) override {
    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    float time =
        std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    UniformBufferObject ubo{};
    F32 cameraPos = 0.0f;
    ubo.Model = glm::mat4(1.0f);
    ubo.Model = glm::translate(ubo.Model, glm::vec3(1.0f, 0.5f, 1.0f));
    ubo.Model = glm::rotate(ubo.Model, glm::radians(55.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.Model = glm::scale(ubo.Model, glm::vec3(0.1f, 0.1f, 0.1f));
    ubo.View = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f),
                           glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.Projection = glm::perspective(
        glm::radians(45.0f),
        static_cast<F32>(vk.SwapchainExtent.width) / static_cast<F32>(vk.SwapchainExtent.height),
        0.1f, 1000.0f);
    ubo.Projection[1][1] *= -1.0f;

    void* data;
    vkMapMemory(vk.Device, UniformBufferMemories[imageIndex], 0, sizeof(ubo), 0, &data);
    memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(vk.Device, UniformBufferMemories[imageIndex]);
  }
};

bool UnlitMaterial::Initialized = false;
U32 UnlitMaterial::RefCount = 0;
VkDescriptorSetLayout UnlitMaterial::DescriptorSetLayout = VK_NULL_HANDLE;
VkPipelineLayout UnlitMaterial::PipelineLayout = VK_NULL_HANDLE;
VkPipeline UnlitMaterial::Pipeline = VK_NULL_HANDLE;

struct LitMaterial : public Material {
  static bool Initialized;
  static U32 RefCount;
  static VkDescriptorSetLayout DescriptorSetLayout;
  static VkPipelineLayout PipelineLayout;
  static VkPipeline Pipeline;

  VkDescriptorPool DescriptorPool = VK_NULL_HANDLE;
  std::vector<VkBuffer> UniformBuffers;
  std::vector<VkDeviceMemory> UniformBufferMemories;
  std::vector<VkDescriptorSet> DescriptorSets;

  LitMaterial() {
    if (!Initialized) {
      StaticInitialize();
    }

    // DescriptorPool
    {
      const VkDescriptorPoolSize poolSizes[] = {{
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,  // type
          vk.SwapchainImageCount              // descriptorCount
      }};

      const VkDescriptorPoolCreateInfo createInfo{
          VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,  // sType
          nullptr,                                        // pNext
          0,                                              // flags
          vk.SwapchainImageCount,                         // maxSets
          ArrLen(poolSizes),                              // poolSizeCount
          poolSizes                                       // pPoolSizes
      };
      VkCall(vkCreateDescriptorPool(vk.Device, &createInfo, nullptr, &DescriptorPool));
    }

    // Uniform Buffers
    {
      VkDeviceSize bufferSize = sizeof(UniformBufferObject);
      UniformBuffers.resize(vk.SwapchainImageCount, VK_NULL_HANDLE);
      UniformBufferMemories.resize(vk.SwapchainImageCount, VK_NULL_HANDLE);

      for (U32 i = 0; i < vk.SwapchainImageCount; i++) {
        CreateBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     UniformBuffers[i], UniformBufferMemories[i]);
      }
    }

    // Descriptor Sets
    {
      std::vector<VkDescriptorSetLayout> layouts(vk.SwapchainImageCount, DescriptorSetLayout);

      const VkDescriptorSetAllocateInfo allocateInfo{
          VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,  // sType
          nullptr,                                         // pNext
          DescriptorPool,                                  // decriptorPool
          vk.SwapchainImageCount,                          // descriptorSetCount
          layouts.data()                                   // pSetLayouts
      };

      DescriptorSets.resize(vk.SwapchainImageCount, VK_NULL_HANDLE);
      VkCall(vkAllocateDescriptorSets(vk.Device, &allocateInfo, DescriptorSets.data()));

      for (U32 i = 0; i < vk.SwapchainImageCount; i++) {
        const VkDescriptorBufferInfo bufferInfo{
            UniformBuffers[i],           // buffer
            0,                           // offset
            sizeof(UniformBufferObject)  // range
        };

        const VkWriteDescriptorSet descriptorWrites[] = {{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,  // sType
            nullptr,                                 // pNext
            DescriptorSets[i],                       // dstSet
            0,                                       // dstBinding
            0,                                       // dstArrayElement
            1,                                       // descriptorCount
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,       // descriptorType
            nullptr,                                 // pImageInfo
            &bufferInfo,                             // pBufferInfo
            nullptr                                  // pTexelBufferView

        }};

        vkUpdateDescriptorSets(vk.Device, ArrLen(descriptorWrites), descriptorWrites, 0, nullptr);
      }
    }
  }

  ~LitMaterial() {
    for (U32 i = 0; i < vk.SwapchainImageCount; i++) {
      vkDestroyBuffer(vk.Device, UniformBuffers[i], nullptr);
      vkFreeMemory(vk.Device, UniformBufferMemories[i], nullptr);
    }
    vkDestroyDescriptorPool(vk.Device, DescriptorPool, nullptr);

    if (--RefCount == 0) {
      StaticDestroy();
    }
  }

  void StaticInitialize() override {
    // DescriptorSetLayout
    {
      constexpr static const VkDescriptorSetLayoutBinding uboLayoutBinding{
          0,                                  // binding
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,  // descriptorType
          1,                                  // descriptorCount
          VK_SHADER_STAGE_VERTEX_BIT,         // stageFlags
          nullptr                             // pImmutableSamplers
      };

      constexpr static const VkDescriptorSetLayoutBinding bindings[] = {uboLayoutBinding};

      constexpr static const VkDescriptorSetLayoutCreateInfo layoutCreateInfo{
          VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,  // sType
          nullptr,                                              // pNext
          0,                                                    // flags
          sizeof(bindings) / sizeof(*bindings),                 // bindingCount
          bindings                                              // pBindings
      };
      VkCall(
          vkCreateDescriptorSetLayout(vk.Device, &layoutCreateInfo, nullptr, &DescriptorSetLayout));
    }

    // Pipeline
    {
      VkShaderModule vertexShaderModule =
          CreateShaderModule(vk.Device, "assets/shaders/Lit.vert.spv");
      VkShaderModule fragmentShaderModule =
          CreateShaderModule(vk.Device, "assets/shaders/Lit.frag.spv");

      constexpr static const VkVertexInputBindingDescription vertexBindingDescription{
          0,                           // binding
          sizeof(Vertex),              // stride
          VK_VERTEX_INPUT_RATE_VERTEX  // inputRate
      };

      constexpr static const VkVertexInputAttributeDescription vertexAttributeDescription[]{
          {
              0,                           // location
              0,                           // binding
              VK_FORMAT_R32G32B32_SFLOAT,  // format
              offsetof(Vertex, Position)   // offset
          },                               // Position
          {
              1,                           // location
              0,                           // binding
              VK_FORMAT_R32G32B32_SFLOAT,  // format
              offsetof(Vertex, Normal)     // offset
          },                               // Normal
          {
              2,                          // location
              0,                          // binding
              VK_FORMAT_R32G32_SFLOAT,    // format
              offsetof(Vertex, TexCoord)  // offset
          },                              // TexCoord
          {
              3,                           // location
              0,                           // binding
              VK_FORMAT_R32G32B32_SFLOAT,  // format
              offsetof(Vertex, Color)      // offset
          },                               // Color
      };

      const VkPipelineShaderStageCreateInfo vertexShaderStageCreateInfo{
          VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,  // sType
          nullptr,                                              // pNext
          0,                                                    // flags
          VK_SHADER_STAGE_VERTEX_BIT,                           // stage
          vertexShaderModule,                                   // module
          "main",                                               // pName
          nullptr                                               // pSpecializationInfo
      };

      const VkPipelineShaderStageCreateInfo fragmentShaderStageCreateInfo{
          VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,  // sType
          nullptr,                                              // pNext
          0,                                                    // flags
          VK_SHADER_STAGE_FRAGMENT_BIT,                         // stage
          fragmentShaderModule,                                 // module
          "main",                                               // pName
          nullptr                                               // pSpecializationInfo
      };

      const VkPipelineShaderStageCreateInfo shaderStageCreateInfos[] = {
          vertexShaderStageCreateInfo, fragmentShaderStageCreateInfo};

      constexpr static const VkPipelineVertexInputStateCreateInfo vertexInputInfo{
          VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,  // sType
          nullptr,                                                    // pNext
          0,                                                          // flags
          1,                          // vertexBindingDescriptionCount
          &vertexBindingDescription,  // pVertexBindingDescriptions
          sizeof(vertexAttributeDescription) /
              sizeof(*vertexAttributeDescription),  // vertexAttributeDescriptionCount
          vertexAttributeDescription                // pVertexAttributeDescriptions
      };

      constexpr static const VkPipelineInputAssemblyStateCreateInfo inputAssembly{
          VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,  // sType
          nullptr,                                                      // pNext
          0,                                                            // flags
          VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,                          // topology
          VK_FALSE                                                      // primitiveRestartEnable
      };

      const VkPipelineViewportStateCreateInfo viewportState{
          VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,  // sType
          nullptr,                                                // pNext
          0,                                                      // flags
          1,                                                      // viewportCount
          nullptr,                                                // pViewports
          1,                                                      // scissorCount
          nullptr                                                 // pScissors
      };

      constexpr static const VkPipelineRasterizationStateCreateInfo rasterizer{
          VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,  // sType
          nullptr,                                                     // pNext
          0,                                                           // flags
          VK_FALSE,                                                    // depthClampEnable
          VK_FALSE,                                                    // rasterizerDiscardEnable
          VK_POLYGON_MODE_FILL,                                        // polygonMode
          VK_CULL_MODE_BACK_BIT,                                       // cullMode
          VK_FRONT_FACE_COUNTER_CLOCKWISE,                             // frontFace
          VK_FALSE,                                                    // depthBiasEnable
          0.0f,                                                        // depthBiasConstantFactor
          0.0f,                                                        // depthBiasClamp
          0.0f,                                                        // depthBiasSlopeFactor
          1.0f                                                         // lineWidth
      };

      constexpr static const VkPipelineMultisampleStateCreateInfo multisampling{
          VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,  // sType
          nullptr,                                                   // pNext
          0,                                                         // flags
          VK_SAMPLE_COUNT_1_BIT,                                     // rasterizationSamples
          VK_FALSE,                                                  // sampleShadingEnable
          1.0f,                                                      // minSampleShading
          nullptr,                                                   // pSampleMask
          VK_FALSE,                                                  // alphaToCoverageEnable
          VK_FALSE                                                   // alphaToOneEnable
      };

      constexpr static const VkPipelineColorBlendAttachmentState colorBlendAttachment{
          VK_TRUE,                              // blendEnable
          VK_BLEND_FACTOR_SRC_ALPHA,            // srcColorBlendFactor
          VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,  // dstColorBlendFactor
          VK_BLEND_OP_ADD,                      // colorBlendOp
          VK_BLEND_FACTOR_ONE,                  // srcAlphaBlendFactor
          VK_BLEND_FACTOR_ZERO,                 // dstAlphaBlendFactor
          VK_BLEND_OP_ADD,                      // alphaBlendOp
          VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
              VK_COLOR_COMPONENT_A_BIT  // colorWriteMask
      };

      constexpr static const VkPipelineColorBlendStateCreateInfo colorBlending{
          VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,  // sType
          nullptr,                                                   // pNext
          0,                                                         // flags
          VK_FALSE,                                                  // logicOpEnable
          VK_LOGIC_OP_COPY,                                          // logicOp
          1,                                                         // attachmentCount
          &colorBlendAttachment,                                     // pAttachment
          {0.0f, 0.0f, 0.0f, 0.0f}                                   // blendConstants
      };

      constexpr static const VkPipelineDepthStencilStateCreateInfo depthStencil{
          VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,  // sType
          nullptr,                                                     // pNext
          0,                                                           // flags
          VK_TRUE,                                                     // depthTestEnable
          VK_TRUE,                                                     // depthWriteEnable
          VK_COMPARE_OP_LESS,                                          // depthCompareOp
          VK_FALSE,                                                    // depthBoundsTestEnable
          VK_FALSE,                                                    // stencilTestEnable
          {},                                                          // front
          {},                                                          // back
          0.0f,                                                        // minDepthBounds
          1.0f                                                         // maxDepthBounds
      };

      std::vector<VkDynamicState> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT,
                                                   VK_DYNAMIC_STATE_SCISSOR};

      const VkPipelineDynamicStateCreateInfo dynamicState{
          VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,  // sType
          nullptr,                                               // pNext
          0,                                                     // flags
          static_cast<U32>(dynamicStates.size()),                // dynamicStateCount
          dynamicStates.data()                                   // pDynamicStates
      };

      const VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{
          VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,  // sType
          nullptr,                                        // pNext
          0,                                              // flags
          1,                                              // setLayoutCount
          &DescriptorSetLayout,                           // pSetLayouts
          0,                                              // pushConstantRangeCount
          nullptr                                         // pPushConstantRanges
      };

      VkCall(
          vkCreatePipelineLayout(vk.Device, &pipelineLayoutCreateInfo, nullptr, &PipelineLayout));

      const VkGraphicsPipelineCreateInfo pipelineCreateInfo{
          VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,  // sType
          nullptr,                                          // pNext
          0,                                                // flags
          2,                                                // stageCount
          shaderStageCreateInfos,                           // pStages
          &vertexInputInfo,                                 // pVertexInputState
          &inputAssembly,                                   // pInputAssemblyState
          nullptr,                                          // pTessellationState
          &viewportState,                                   // pViewportState
          &rasterizer,                                      // pRasterizationState
          &multisampling,                                   // pMultisampleState
          &depthStencil,                                    // pDepthStencilState
          &colorBlending,                                   // pColorBlendState
          &dynamicState,                                    // pDynamicState
          PipelineLayout,                                   // layout
          vk.RenderPass,                                    // renderPass
          0,                                                // subpass,
          VK_NULL_HANDLE,                                   // basePipelineHandle
          -1                                                // basePipelineIndex
      };

      VkCall(vkCreateGraphicsPipelines(vk.Device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr,
                                       &Pipeline));

      vkDestroyShaderModule(vk.Device, fragmentShaderModule, nullptr);
      vkDestroyShaderModule(vk.Device, vertexShaderModule, nullptr);
    }

    RefCount = 1;
    Initialized = true;
  }

  void StaticDestroy() override {
    vkDestroyPipeline(vk.Device, Pipeline, nullptr);
    vkDestroyPipelineLayout(vk.Device, PipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(vk.Device, DescriptorSetLayout, nullptr);

    Pipeline = VK_NULL_HANDLE;
    PipelineLayout = VK_NULL_HANDLE;
    DescriptorSetLayout = VK_NULL_HANDLE;
    Initialized = false;
    RefCount = 0;
  }

  void Bind(VkCommandBuffer& cmdBuf, U32 imageIndex) override {
    // vkCmdBindPipeline
    { vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, Pipeline); }

    // Update viewport/scissor
    vkCmdSetViewport(cmdBuf, 0, 1, &vk.Viewport);
    vkCmdSetScissor(cmdBuf, 0, 1, &vk.Scissor);

    // Bind descriptor set
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, PipelineLayout, 0, 1,
                            &DescriptorSets[imageIndex], 0, nullptr);
  }

  void Update(U32 imageIndex) override {
    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    float time =
        std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    UniformBufferObject ubo{};
    F32 cameraPos = 0.0f;
    ubo.Model = glm::mat4(1.0f);
    ubo.Model = glm::rotate(ubo.Model, glm::radians(15.0f) + time, glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.Model = glm::scale(ubo.Model, glm::vec3(0.7f, 0.7f, 0.7f));
    ubo.View = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f),
                           glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.Projection = glm::perspective(
        glm::radians(45.0f),
        static_cast<F32>(vk.SwapchainExtent.width) / static_cast<F32>(vk.SwapchainExtent.height),
        0.1f, 1000.0f);
    ubo.Projection[1][1] *= -1.0f;

    void* data;
    vkMapMemory(vk.Device, UniformBufferMemories[imageIndex], 0, sizeof(ubo), 0, &data);
    memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(vk.Device, UniformBufferMemories[imageIndex]);
  }
};

bool LitMaterial::Initialized = false;
U32 LitMaterial::RefCount = 0;
VkDescriptorSetLayout LitMaterial::DescriptorSetLayout = VK_NULL_HANDLE;
VkPipelineLayout LitMaterial::PipelineLayout = VK_NULL_HANDLE;
VkPipeline LitMaterial::Pipeline = VK_NULL_HANDLE;

struct UnlitTexturedMaterial : public Material {
  static bool Initialized;
  static U32 RefCount;
  static VkDescriptorSetLayout DescriptorSetLayout;
  static VkPipelineLayout PipelineLayout;
  static VkPipeline Pipeline;

  VkDescriptorPool DescriptorPool = VK_NULL_HANDLE;
  std::vector<VkBuffer> UniformBuffers;
  std::vector<VkDeviceMemory> UniformBufferMemories;
  VkImage TextureImage = VK_NULL_HANDLE;
  VkDeviceMemory TextureImageMemory = VK_NULL_HANDLE;
  VkImageView TextureImageView = VK_NULL_HANDLE;
  VkSampler TextureSampler = VK_NULL_HANDLE;
  std::vector<VkDescriptorSet> DescriptorSets;

  UnlitTexturedMaterial(const std::string& textureFile) {
    if (!Initialized) {
      StaticInitialize();
    }

    // DescriptorPool
    {
      const VkDescriptorPoolSize poolSizes[] = {{
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,  // type
          vk.SwapchainImageCount              // descriptorCount
      }};

      const VkDescriptorPoolCreateInfo createInfo{
          VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,  // sType
          nullptr,                                        // pNext
          0,                                              // flags
          vk.SwapchainImageCount,                         // maxSets
          ArrLen(poolSizes),                              // poolSizeCount
          poolSizes                                       // pPoolSizes
      };
      VkCall(vkCreateDescriptorPool(vk.Device, &createInfo, nullptr, &DescriptorPool));
    }

    // Uniform Buffers
    {
      VkDeviceSize bufferSize = sizeof(UniformBufferObject);
      UniformBuffers.resize(vk.SwapchainImageCount, VK_NULL_HANDLE);
      UniformBufferMemories.resize(vk.SwapchainImageCount, VK_NULL_HANDLE);

      for (U32 i = 0; i < vk.SwapchainImageCount; i++) {
        CreateBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     UniformBuffers[i], UniformBufferMemories[i]);
      }
    }

    // Texture
    {
      I32 textureWidth, textureHeight, textureChannels;
      stbi_uc* texturePixels = stbi_load(textureFile.c_str(), &textureWidth, &textureHeight,
                                         &textureChannels, STBI_rgb_alpha);
      if (!texturePixels) {
        throw std::runtime_error("Failed to load texture image!");
      }

      VkDeviceSize imageBufferSize = textureWidth * textureHeight * 4ll;
      VkBuffer stagingBuffer;
      VkDeviceMemory stagingDeviceMemory;
      ASSERT(
          CreateBuffer(imageBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       stagingBuffer, stagingDeviceMemory));
      void* data;
      vkMapMemory(vk.Device, stagingDeviceMemory, 0, imageBufferSize, 0, &data);
      memcpy(data, texturePixels, static_cast<size_t>(imageBufferSize));
      vkUnmapMemory(vk.Device, stagingDeviceMemory);

      ASSERT(CreateImage(textureWidth, textureHeight, VK_FORMAT_R8G8B8A8_SRGB,
                         VK_IMAGE_TILING_OPTIMAL,
                         VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, TextureImage, TextureImageMemory));
      TransitionImageLayout(TextureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
      CopyBufferToImage(stagingBuffer, TextureImage, static_cast<U32>(textureWidth),
                        static_cast<U32>(textureHeight));
      TransitionImageLayout(TextureImage, VK_FORMAT_R8G8B8A8_SRGB,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

      vkDestroyBuffer(vk.Device, stagingBuffer, nullptr);
      vkFreeMemory(vk.Device, stagingDeviceMemory, nullptr);

      const VkImageViewCreateInfo createInfo{
          VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,  // sType
          nullptr,                                   // pNext
          0,                                         // flags
          TextureImage,                              // image
          VK_IMAGE_VIEW_TYPE_2D,                     // viewType
          VK_FORMAT_R8G8B8A8_SRGB,                   // format
          {
              VK_COMPONENT_SWIZZLE_IDENTITY,  // components.r
              VK_COMPONENT_SWIZZLE_IDENTITY,  // components.g
              VK_COMPONENT_SWIZZLE_IDENTITY   // components.b
          },                                  // components
          {
              VK_IMAGE_ASPECT_COLOR_BIT,  // subresourceRange.aspectMask
              0,                          // subresourceRange.baseMipLevel
              1,                          // subresourceRange.levelCount
              0,                          // subresourceRange.baseArrayLayer
              1                           // subresourceRange.layerCount
          }                               // subresourceRange
      };
      VkCall(vkCreateImageView(vk.Device, &createInfo, nullptr, &TextureImageView));
    }

    // Texture Samplers
    {
      constexpr static const VkSamplerCreateInfo createInfo{
          VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,  // sType
          nullptr,                                // pNext
          0,                                      // flags
          VK_FILTER_LINEAR,                       // magFilter
          VK_FILTER_LINEAR,                       // minFilter
          VK_SAMPLER_MIPMAP_MODE_LINEAR,          // mipmapMode
          VK_SAMPLER_ADDRESS_MODE_REPEAT,         // addressModeU
          VK_SAMPLER_ADDRESS_MODE_REPEAT,         // addressModeV
          VK_SAMPLER_ADDRESS_MODE_REPEAT,         // addressModeW
          0.0f,                                   // mipLodBias
          VK_TRUE,                                // anisotropyEnable
          16.0f,                                  // maxAnisotropy
          VK_FALSE,                               // compareEnable
          VK_COMPARE_OP_ALWAYS,                   // compareOp
          0.0f,                                   // minLod
          0.0f,                                   // maxLod
          VK_BORDER_COLOR_INT_OPAQUE_BLACK,       // borderColor
          VK_FALSE                                // unnormalizedCoordinates
      };
      VkCall(vkCreateSampler(vk.Device, &createInfo, nullptr, &TextureSampler));
    }

    // Descriptor Sets
    {
      std::vector<VkDescriptorSetLayout> layouts(vk.SwapchainImageCount, DescriptorSetLayout);

      const VkDescriptorSetAllocateInfo allocateInfo{
          VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,  // sType
          nullptr,                                         // pNext
          DescriptorPool,                                  // decriptorPool
          vk.SwapchainImageCount,                          // descriptorSetCount
          layouts.data()                                   // pSetLayouts
      };

      DescriptorSets.resize(vk.SwapchainImageCount, VK_NULL_HANDLE);
      VkCall(vkAllocateDescriptorSets(vk.Device, &allocateInfo, DescriptorSets.data()));

      for (U32 i = 0; i < vk.SwapchainImageCount; i++) {
        const VkDescriptorBufferInfo bufferInfo{
            UniformBuffers[i],           // buffer
            0,                           // offset
            sizeof(UniformBufferObject)  // range
        };

        const VkDescriptorImageInfo imageInfo{
            TextureSampler,                           // sampler
            TextureImageView,                         // imageView
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL  // imageLayout
        };

        const VkWriteDescriptorSet descriptorWrites[] = {
            {
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,  // sType
                nullptr,                                 // pNext
                DescriptorSets[i],                       // dstSet
                0,                                       // dstBinding
                0,                                       // dstArrayElement
                1,                                       // descriptorCount
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,       // descriptorType
                nullptr,                                 // pImageInfo
                &bufferInfo,                             // pBufferInfo
                nullptr                                  // pTexelBufferView

            },
            {
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,     // sType
                nullptr,                                    // pNext
                DescriptorSets[i],                          // dstSet
                1,                                          // dstBinding
                0,                                          // dstArrayElement
                1,                                          // descriptorCount
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,  // descriptorType
                &imageInfo,                                 // pImageInfo
                nullptr,                                    // pBufferInfo
                nullptr                                     // pTexelBufferView

            }};

        vkUpdateDescriptorSets(vk.Device, ArrLen(descriptorWrites), descriptorWrites, 0, nullptr);
      }
    }
  }

  ~UnlitTexturedMaterial() {
    vkDestroySampler(vk.Device, TextureSampler, nullptr);
    vkDestroyImageView(vk.Device, TextureImageView, nullptr);
    vkDestroyImage(vk.Device, TextureImage, nullptr);
    vkFreeMemory(vk.Device, TextureImageMemory, nullptr);
    for (U32 i = 0; i < vk.SwapchainImageCount; i++) {
      vkDestroyBuffer(vk.Device, UniformBuffers[i], nullptr);
      vkFreeMemory(vk.Device, UniformBufferMemories[i], nullptr);
    }
    vkDestroyDescriptorPool(vk.Device, DescriptorPool, nullptr);

    if (--RefCount == 0) {
      StaticDestroy();
    }
  }

  void StaticInitialize() override {
    // DescriptorSetLayout
    {
      constexpr static const VkDescriptorSetLayoutBinding uboLayoutBinding{
          0,                                  // binding
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,  // descriptorType
          1,                                  // descriptorCount
          VK_SHADER_STAGE_VERTEX_BIT,         // stageFlags
          nullptr                             // pImmutableSamplers
      };

      constexpr static const VkDescriptorSetLayoutBinding samplerLayoutBinding{
          1,                                          // binding
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,  // descriptorType
          1,                                          // descriptorCount
          VK_SHADER_STAGE_FRAGMENT_BIT,               // stageFlags
          nullptr                                     // pImmutableSamplers
      };

      constexpr static const VkDescriptorSetLayoutBinding bindings[] = {uboLayoutBinding,
                                                                        samplerLayoutBinding};

      constexpr static const VkDescriptorSetLayoutCreateInfo layoutCreateInfo{
          VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,  // sType
          nullptr,                                              // pNext
          0,                                                    // flags
          sizeof(bindings) / sizeof(*bindings),                 // bindingCount
          bindings                                              // pBindings
      };
      VkCall(
          vkCreateDescriptorSetLayout(vk.Device, &layoutCreateInfo, nullptr, &DescriptorSetLayout));
    }

    // Pipeline
    {
      VkShaderModule vertexShaderModule =
          CreateShaderModule(vk.Device, "assets/shaders/Basic.vert.spv");
      VkShaderModule fragmentShaderModule =
          CreateShaderModule(vk.Device, "assets/shaders/Basic.frag.spv");

      constexpr static const VkVertexInputBindingDescription vertexBindingDescription{
          0,                           // binding
          sizeof(Vertex),              // stride
          VK_VERTEX_INPUT_RATE_VERTEX  // inputRate
      };

      constexpr static const VkVertexInputAttributeDescription vertexAttributeDescription[]{
          {
              0,                           // location
              0,                           // binding
              VK_FORMAT_R32G32B32_SFLOAT,  // format
              offsetof(Vertex, Position)   // offset
          },                               // Position
          {
              1,                           // location
              0,                           // binding
              VK_FORMAT_R32G32B32_SFLOAT,  // format
              offsetof(Vertex, Normal)     // offset
          },                               // Normal
          {
              2,                          // location
              0,                          // binding
              VK_FORMAT_R32G32_SFLOAT,    // format
              offsetof(Vertex, TexCoord)  // offset
          },                              // TexCoord
          {
              3,                           // location
              0,                           // binding
              VK_FORMAT_R32G32B32_SFLOAT,  // format
              offsetof(Vertex, Color)      // offset
          },                               // Color
      };

      const VkPipelineShaderStageCreateInfo vertexShaderStageCreateInfo{
          VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,  // sType
          nullptr,                                              // pNext
          0,                                                    // flags
          VK_SHADER_STAGE_VERTEX_BIT,                           // stage
          vertexShaderModule,                                   // module
          "main",                                               // pName
          nullptr                                               // pSpecializationInfo
      };

      const VkPipelineShaderStageCreateInfo fragmentShaderStageCreateInfo{
          VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,  // sType
          nullptr,                                              // pNext
          0,                                                    // flags
          VK_SHADER_STAGE_FRAGMENT_BIT,                         // stage
          fragmentShaderModule,                                 // module
          "main",                                               // pName
          nullptr                                               // pSpecializationInfo
      };

      const VkPipelineShaderStageCreateInfo shaderStageCreateInfos[] = {
          vertexShaderStageCreateInfo, fragmentShaderStageCreateInfo};

      constexpr static const VkPipelineVertexInputStateCreateInfo vertexInputInfo{
          VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,  // sType
          nullptr,                                                    // pNext
          0,                                                          // flags
          1,                          // vertexBindingDescriptionCount
          &vertexBindingDescription,  // pVertexBindingDescriptions
          sizeof(vertexAttributeDescription) /
              sizeof(*vertexAttributeDescription),  // vertexAttributeDescriptionCount
          vertexAttributeDescription                // pVertexAttributeDescriptions
      };

      constexpr static const VkPipelineInputAssemblyStateCreateInfo inputAssembly{
          VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,  // sType
          nullptr,                                                      // pNext
          0,                                                            // flags
          VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,                          // topology
          VK_FALSE                                                      // primitiveRestartEnable
      };

      const VkPipelineViewportStateCreateInfo viewportState{
          VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,  // sType
          nullptr,                                                // pNext
          0,                                                      // flags
          1,                                                      // viewportCount
          nullptr,                                                // pViewports
          1,                                                      // scissorCount
          nullptr                                                 // pScissors
      };

      constexpr static const VkPipelineRasterizationStateCreateInfo rasterizer{
          VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,  // sType
          nullptr,                                                     // pNext
          0,                                                           // flags
          VK_FALSE,                                                    // depthClampEnable
          VK_FALSE,                                                    // rasterizerDiscardEnable
          VK_POLYGON_MODE_FILL,                                        // polygonMode
          VK_CULL_MODE_BACK_BIT,                                       // cullMode
          VK_FRONT_FACE_COUNTER_CLOCKWISE,                             // frontFace
          VK_FALSE,                                                    // depthBiasEnable
          0.0f,                                                        // depthBiasConstantFactor
          0.0f,                                                        // depthBiasClamp
          0.0f,                                                        // depthBiasSlopeFactor
          1.0f                                                         // lineWidth
      };

      constexpr static const VkPipelineMultisampleStateCreateInfo multisampling{
          VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,  // sType
          nullptr,                                                   // pNext
          0,                                                         // flags
          VK_SAMPLE_COUNT_1_BIT,                                     // rasterizationSamples
          VK_FALSE,                                                  // sampleShadingEnable
          1.0f,                                                      // minSampleShading
          nullptr,                                                   // pSampleMask
          VK_FALSE,                                                  // alphaToCoverageEnable
          VK_FALSE                                                   // alphaToOneEnable
      };

      constexpr static const VkPipelineColorBlendAttachmentState colorBlendAttachment{
          VK_TRUE,                              // blendEnable
          VK_BLEND_FACTOR_SRC_ALPHA,            // srcColorBlendFactor
          VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,  // dstColorBlendFactor
          VK_BLEND_OP_ADD,                      // colorBlendOp
          VK_BLEND_FACTOR_ONE,                  // srcAlphaBlendFactor
          VK_BLEND_FACTOR_ZERO,                 // dstAlphaBlendFactor
          VK_BLEND_OP_ADD,                      // alphaBlendOp
          VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
              VK_COLOR_COMPONENT_A_BIT  // colorWriteMask
      };

      constexpr static const VkPipelineColorBlendStateCreateInfo colorBlending{
          VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,  // sType
          nullptr,                                                   // pNext
          0,                                                         // flags
          VK_FALSE,                                                  // logicOpEnable
          VK_LOGIC_OP_COPY,                                          // logicOp
          1,                                                         // attachmentCount
          &colorBlendAttachment,                                     // pAttachment
          {0.0f, 0.0f, 0.0f, 0.0f}                                   // blendConstants
      };

      constexpr static const VkPipelineDepthStencilStateCreateInfo depthStencil{
          VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,  // sType
          nullptr,                                                     // pNext
          0,                                                           // flags
          VK_TRUE,                                                     // depthTestEnable
          VK_TRUE,                                                     // depthWriteEnable
          VK_COMPARE_OP_LESS,                                          // depthCompareOp
          VK_FALSE,                                                    // depthBoundsTestEnable
          VK_FALSE,                                                    // stencilTestEnable
          {},                                                          // front
          {},                                                          // back
          0.0f,                                                        // minDepthBounds
          1.0f                                                         // maxDepthBounds
      };

      std::vector<VkDynamicState> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT,
                                                   VK_DYNAMIC_STATE_SCISSOR};

      const VkPipelineDynamicStateCreateInfo dynamicState{
          VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,  // sType
          nullptr,                                               // pNext
          0,                                                     // flags
          static_cast<U32>(dynamicStates.size()),                // dynamicStateCount
          dynamicStates.data()                                   // pDynamicStates
      };

      const VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{
          VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,  // sType
          nullptr,                                        // pNext
          0,                                              // flags
          1,                                              // setLayoutCount
          &DescriptorSetLayout,                           // pSetLayouts
          0,                                              // pushConstantRangeCount
          nullptr                                         // pPushConstantRanges
      };

      VkCall(
          vkCreatePipelineLayout(vk.Device, &pipelineLayoutCreateInfo, nullptr, &PipelineLayout));

      const VkGraphicsPipelineCreateInfo pipelineCreateInfo{
          VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,  // sType
          nullptr,                                          // pNext
          0,                                                // flags
          2,                                                // stageCount
          shaderStageCreateInfos,                           // pStages
          &vertexInputInfo,                                 // pVertexInputState
          &inputAssembly,                                   // pInputAssemblyState
          nullptr,                                          // pTessellationState
          &viewportState,                                   // pViewportState
          &rasterizer,                                      // pRasterizationState
          &multisampling,                                   // pMultisampleState
          &depthStencil,                                    // pDepthStencilState
          &colorBlending,                                   // pColorBlendState
          &dynamicState,                                    // pDynamicState
          PipelineLayout,                                   // layout
          vk.RenderPass,                                    // renderPass
          0,                                                // subpass,
          VK_NULL_HANDLE,                                   // basePipelineHandle
          -1                                                // basePipelineIndex
      };

      VkCall(vkCreateGraphicsPipelines(vk.Device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr,
                                       &Pipeline));

      vkDestroyShaderModule(vk.Device, fragmentShaderModule, nullptr);
      vkDestroyShaderModule(vk.Device, vertexShaderModule, nullptr);
    }

    RefCount = 1;
    Initialized = true;
  }

  void StaticDestroy() override {
    vkDestroyPipeline(vk.Device, Pipeline, nullptr);
    vkDestroyPipelineLayout(vk.Device, PipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(vk.Device, DescriptorSetLayout, nullptr);

    Pipeline = VK_NULL_HANDLE;
    PipelineLayout = VK_NULL_HANDLE;
    DescriptorSetLayout = VK_NULL_HANDLE;
    Initialized = false;
    RefCount = 0;
  }

  void Bind(VkCommandBuffer& cmdBuf, U32 imageIndex) override {
    // vkCmdBindPipeline
    { vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, Pipeline); }

    // Update viewport/scissor
    vkCmdSetViewport(cmdBuf, 0, 1, &vk.Viewport);
    vkCmdSetScissor(cmdBuf, 0, 1, &vk.Scissor);

    // Bind descriptor set
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, PipelineLayout, 0, 1,
                            &DescriptorSets[imageIndex], 0, nullptr);
  }

  void Update(U32 imageIndex) override {
    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    float time =
        std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    UniformBufferObject ubo{};
    F32 cameraPos = 0.0f;
    ubo.Model = glm::mat4(1.0f);
    ubo.Model = glm::rotate(ubo.Model, glm::radians(25.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    // ubo.Model = glm::scale(ubo.Model, glm::vec3(10.0f, 10.0f, 10.0f));
    ubo.View = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f),
                           glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.Projection = glm::perspective(
        glm::radians(45.0f),
        static_cast<F32>(vk.SwapchainExtent.width) / static_cast<F32>(vk.SwapchainExtent.height),
        0.1f, 1000.0f);
    ubo.Projection[1][1] *= -1.0f;

    void* data;
    vkMapMemory(vk.Device, UniformBufferMemories[imageIndex], 0, sizeof(ubo), 0, &data);
    memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(vk.Device, UniformBufferMemories[imageIndex]);
  }
};

bool UnlitTexturedMaterial::Initialized = false;
U32 UnlitTexturedMaterial::RefCount = 0;
VkDescriptorSetLayout UnlitTexturedMaterial::DescriptorSetLayout = VK_NULL_HANDLE;
VkPipelineLayout UnlitTexturedMaterial::PipelineLayout = VK_NULL_HANDLE;
VkPipeline UnlitTexturedMaterial::Pipeline = VK_NULL_HANDLE;

struct Object {
  Mesh* ObjectMesh;
  Material* ObjectMaterial;
  glm::mat4 Transform;
  bool Uploaded = false;

  Object(Mesh* mesh, Material* material) {
    ObjectMesh = mesh;
    ObjectMaterial = material;
    Transform = glm::mat4(1.0f);
  }

  ~Object() {
    if (Uploaded) {
      FreeMesh(*ObjectMesh);
      Uploaded = false;
    }
    delete ObjectMesh;
    delete ObjectMaterial;
  }

  void Enable() {
    if (!Uploaded) {
      UploadMesh(*ObjectMesh);
      Uploaded = true;
    }
    vk.ObjectsToRender.push_back(this);
  }

  void Disable() {
    FreeMesh(*ObjectMesh);
    vk.ObjectsToRender.remove(this);
  }
};

// ======================================================================
// Functions
// ======================================================================
template <typename T>
inline T Min(T a, T b) {
  return a <= b ? a : b;
}

template <typename T>
inline T Max(T a, T b) {
  return a >= b ? a : b;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL
VulkanDebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                    VkDebugUtilsMessageTypeFlagsEXT messageType,
                    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
  switch (messageSeverity) {
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
      LogError("%s", pCallbackData->pMessage);
      break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
      LogWarn("%s", pCallbackData->pMessage);
      break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
      LogInfo("%s", pCallbackData->pMessage);
      break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
      LogTrace("%s", pCallbackData->pMessage);
      break;
  }

  return VK_FALSE;
}

LRESULT CALLBACK ApplicationWindowProcedure(HWND hwnd, U32 msg, WPARAM wParam, LPARAM lParam) {
  switch (msg) {
    case WM_CLOSE:
      app.CloseRequested = true;
      return true;
    case WM_DESTROY:
      LogFatal("Application window destroyed. Shutting down...");
      PostQuitMessage(0);
      return true;
    case WM_SETCURSOR:
      // Only set cursor if it's in our rendering area
      if (LOWORD(lParam) == HTCLIENT) {
        SetCursor(app.Cursor);
        return true;
      }
  }

  return DefWindowProcW(hwnd, msg, wParam, lParam);
}

VkShaderModule CreateShaderModule(VkDevice vkDevice, const std::string& sourceFile) {
  std::ifstream file(sourceFile, std::ios::ate | std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file for reading!");
  }

  size_t fileSize = file.tellg();
  std::string source;
  source.reserve(fileSize);
  file.seekg(0, std::ios::beg);
  source.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  file.close();

  const VkShaderModuleCreateInfo createInfo{
      VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,  // sType
      nullptr,                                      // pNext
      0,                                            // flags
      static_cast<U32>(source.size()),              // codeSize
      reinterpret_cast<const U32*>(source.data())   // pCode
  };

  VkShaderModule module;
  VkCall(vkCreateShaderModule(vkDevice, &createInfo, nullptr, &module));
  return module;
}

bool CreateImage(U32 width, U32 height, VkFormat format, VkImageTiling tiling,
                 VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image,
                 VkDeviceMemory& memory) {
  const VkImageCreateInfo createInfo{
      VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,  // sType
      nullptr,                              // pNext
      0,                                    // flags
      VK_IMAGE_TYPE_2D,                     // imageType
      format,                               // format
      {
          width,   // extent.width
          height,  // extent.height
          1        // extent.depth
      },
      1,                          // mipLevels
      1,                          // arrayLayers
      VK_SAMPLE_COUNT_1_BIT,      // samples
      tiling,                     // tiling
      usage,                      // usage
      VK_SHARING_MODE_EXCLUSIVE,  // sharingMode
      0,                          // queueFamilyIndexCount
      nullptr,                    // pQueueFamilyIndices
      VK_IMAGE_LAYOUT_UNDEFINED   // initialLayout
  };
  VkCall(vkCreateImage(vk.Device, &createInfo, nullptr, &image));

  VkMemoryRequirements memoryRequirements;
  vkGetImageMemoryRequirements(vk.Device, image, &memoryRequirements);

  const VkMemoryAllocateInfo allocateInfo{
      VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,                        // sType
      nullptr,                                                       // pNext
      memoryRequirements.size,                                       // allocationSize,
      FindMemoryType(memoryRequirements.memoryTypeBits, properties)  // memoryTypeIndex
  };
  VkCall(vkAllocateMemory(vk.Device, &allocateInfo, nullptr, &memory));

  vkBindImageMemory(vk.Device, image, memory, 0);
}

void TransitionImageLayout(VkImage image, VkFormat format, VkImageLayout srcLayout,
                           VkImageLayout dstLayout) {
  const VkCommandBufferAllocateInfo allocateInfo{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,  // sType
      nullptr,                                         // pNext
      vk.GraphicsCommandPool,                          // commandPool
      VK_COMMAND_BUFFER_LEVEL_PRIMARY,                 // level
      1                                                // commandBufferCount
  };

  VkCommandBuffer cmdBuf;
  vkAllocateCommandBuffers(vk.Device, &allocateInfo, &cmdBuf);

  constexpr static const VkCommandBufferBeginInfo beginInfo{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,  // sType
      nullptr,                                      // pNext
      VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,  // flags
      nullptr                                       // pInheritanceInfo
  };
  VkCall(vkBeginCommandBuffer(cmdBuf, &beginInfo));

  VkPipelineStageFlags sourceStage;
  VkPipelineStageFlags destinationStage;

  VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  barrier.oldLayout = srcLayout;
  barrier.newLayout = dstLayout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;
  barrier.srcAccessMask = 0;
  barrier.dstAccessMask = 0;

  if (dstLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT) {
      barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }
  }

  if (srcLayout == VK_IMAGE_LAYOUT_UNDEFINED && dstLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  } else if (srcLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
             dstLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  } else if (srcLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
             dstLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask =
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  } else {
    throw std::runtime_error("Invalid parameters given to TransitionImageLayout!");
  }

  vkCmdPipelineBarrier(cmdBuf, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1,
                       &barrier);

  vkEndCommandBuffer(cmdBuf);

  VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &cmdBuf;

  // TODO: Other queues
  vkQueueSubmit(vk.GraphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(vk.GraphicsQueue);

  vkFreeCommandBuffers(vk.Device, vk.GraphicsCommandPool, 1, &cmdBuf);
}

void CreateSwapchain() {
  // Get swapchain extent
  VkCall(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
      vk.PhysicalDevice, vk.Surface, &vk.PhysicalDeviceInfo.SwapchainSupport.Capabilities));
  const VkSurfaceCapabilitiesKHR& capabilities =
      vk.PhysicalDeviceInfo.SwapchainSupport.Capabilities;
  if (capabilities.currentExtent.width != U32_MAX) {
    vk.SwapchainExtent = capabilities.currentExtent;
  }

  RECT windowRect;
  GetWindowRect(app.Window, &windowRect);
  const U32 appWidth = windowRect.right - windowRect.left;
  const U32 appHeight = windowRect.bottom - windowRect.top;
  vk.SwapchainExtent.width =
      Max(capabilities.minImageExtent.width, Min(capabilities.maxImageExtent.width, appWidth));
  vk.SwapchainExtent.height =
      Max(capabilities.minImageExtent.height, Min(capabilities.maxImageExtent.height, appHeight));
  vk.SwapchainImageCount = Min(vk.SwapchainImageCount,
                               vk.PhysicalDeviceInfo.SwapchainSupport.Capabilities.maxImageCount);

  vk.Viewport = {
      0.0f,                                         // x
      0.0f,                                         // y
      static_cast<F32>(vk.SwapchainExtent.width),   // width
      static_cast<F32>(vk.SwapchainExtent.height),  // height
      0.0f,                                         // minDepth
      1.0f                                          // maxDepth
  };

  vk.Scissor = {
      {0, 0},             // offset
      vk.SwapchainExtent  // extent
  };

  // Determine image sharing mode
  std::set<U32> uniqueQueueFamilyIndices;
  uniqueQueueFamilyIndices.insert(vk.PhysicalDeviceInfo.Queues.GraphicsIndex);
  uniqueQueueFamilyIndices.insert(vk.PhysicalDeviceInfo.Queues.TransferIndex);
  uniqueQueueFamilyIndices.insert(vk.PhysicalDeviceInfo.Queues.PresentationIndex);

  const std::vector<U32> queueFamilyIndices(uniqueQueueFamilyIndices.begin(),
                                            uniqueQueueFamilyIndices.end());
  // If more than one queue needs access to the swapchain, we must state that
  const bool exclusive = queueFamilyIndices.size() == 1;

  const VkSharingMode imageSharingMode =
      exclusive ? VK_SHARING_MODE_EXCLUSIVE : VK_SHARING_MODE_CONCURRENT;
  const U32 queueFamilyIndexCount = exclusive ? 0 : static_cast<U32>(queueFamilyIndices.size());
  const U32* pQueueFamilyIndices = exclusive ? nullptr : queueFamilyIndices.data();

  const VkSwapchainCreateInfoKHR createInfo{
      VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,  // sType
      nullptr,                                      // pNext
      0,                                            // flags
      vk.Surface,                                   // surface
      vk.SwapchainImageCount,                       // minImageCount
      vk.SurfaceFormat.format,                      // imageFormat
      vk.SurfaceFormat.colorSpace,                  // imageColorSpace
      vk.SwapchainExtent,                           // imageExtent
      1,                                            // imageArrayLayers
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,          // imageUsage
      imageSharingMode,                             // imageSharingMode
      queueFamilyIndexCount,                        // queueFamilyIndexCount
      pQueueFamilyIndices,                          // pQueueFamilyIndices
      vk.PhysicalDeviceInfo.SwapchainSupport.Capabilities.currentTransform,  // preTransform
      VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,                                     // compositeAlpha
      vk.PresentMode,                                                        // presentMode
      VK_TRUE,                                                               // clipped
      vk.Swapchain                                                           // oldSwapchain
  };

  VkSwapchainKHR newSwapchain = VK_NULL_HANDLE;
  VkCall(vkCreateSwapchainKHR(vk.Device, &createInfo, nullptr, &newSwapchain));

  if (vk.Swapchain != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(vk.Device);
    vkDestroyImageView(vk.Device, vk.DepthImageView, nullptr);
    vkDestroyImage(vk.Device, vk.DepthImage, nullptr);
    vkFreeMemory(vk.Device, vk.DepthImageMemory, nullptr);
    for (U32 i = 0; i < vk.SwapchainImageCount; i++) {
      vkDestroyFramebuffer(vk.Device, vk.Framebuffers[i], nullptr);
      vkDestroyImageView(vk.Device, vk.SwapchainImageViews[i], nullptr);
    }
    vkDestroySwapchainKHR(vk.Device, vk.Swapchain, nullptr);
  }

  vk.Swapchain = newSwapchain;

  // Create our swapchain images and image views
  VkCall(vkGetSwapchainImagesKHR(vk.Device, vk.Swapchain, &vk.SwapchainImageCount, nullptr));
  vk.SwapchainImages.resize(vk.SwapchainImageCount);
  vk.SwapchainImageViews.resize(vk.SwapchainImageCount);
  VkCall(vkGetSwapchainImagesKHR(vk.Device, vk.Swapchain, &vk.SwapchainImageCount,
                                 vk.SwapchainImages.data()));

  for (U32 i = 0; i < vk.SwapchainImageCount; i++) {
    const VkImageViewCreateInfo createInfo{
        VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,  // sType
        nullptr,                                   // pNext
        0,                                         // flags
        vk.SwapchainImages[i],                     // image
        VK_IMAGE_VIEW_TYPE_2D,                     // viewType
        vk.SurfaceFormat.format,                   // format
        {
            VK_COMPONENT_SWIZZLE_IDENTITY,  // components.r
            VK_COMPONENT_SWIZZLE_IDENTITY,  // components.g
            VK_COMPONENT_SWIZZLE_IDENTITY   // components.b
        },                                  // components
        {
            VK_IMAGE_ASPECT_COLOR_BIT,  // subresourceRange.aspectMask
            0,                          // subresourceRange.baseMipLevel
            1,                          // subresourceRange.levelCount
            0,                          // subresourceRange.baseArrayLayer
            1                           // subresourceRange.layerCount
        }                               // subresourceRange
    };

    VkCall(vkCreateImageView(vk.Device, &createInfo, nullptr, &vk.SwapchainImageViews[i]));
  }

  {
    CreateImage(vk.SwapchainExtent.width, vk.SwapchainExtent.height, vk.DepthFormat,
                VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vk.DepthImage, vk.DepthImageMemory);

    const VkImageViewCreateInfo imageViewCreateInfo{
        VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,  // sType
        nullptr,                                   // pNext
        0,                                         // flags
        vk.DepthImage,                             // image
        VK_IMAGE_VIEW_TYPE_2D,                     // viewType
        vk.DepthFormat,                            // format
        {
            VK_COMPONENT_SWIZZLE_IDENTITY,  // components.r
            VK_COMPONENT_SWIZZLE_IDENTITY,  // components.g
            VK_COMPONENT_SWIZZLE_IDENTITY,  // components.b
            VK_COMPONENT_SWIZZLE_IDENTITY   // components.a
        },
        {
            VK_IMAGE_ASPECT_DEPTH_BIT,  // subresourceRange.aspectMask
            0,                          // subresourceRange.baseMipLevel
            1,                          // subresourceRange.levelCount
            0,                          // subresourceRange.baseArrayLayer
            1                           // subresourceRange.layerCount
        }};
    VkCall(vkCreateImageView(vk.Device, &imageViewCreateInfo, nullptr, &vk.DepthImageView));

    TransitionImageLayout(vk.DepthImage, vk.DepthFormat, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
  }

  vk.Framebuffers.resize(vk.SwapchainImageCount);

  for (U32 i = 0; i < vk.SwapchainImageCount; i++) {
    const VkImageView attachments[] = {vk.SwapchainImageViews[i], vk.DepthImageView};

    const VkFramebufferCreateInfo framebufferCreateInfo{
        VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,  // sType,
        nullptr,                                    // pNext
        0,                                          // flags
        vk.RenderPass,                              // renderPass
        ArrLen(attachments),                        // attachmentCount
        attachments,                                // pAttachments
        vk.SwapchainExtent.width,                   // width
        vk.SwapchainExtent.height,                  // height
        1                                           // layers
    };

    VkCall(vkCreateFramebuffer(vk.Device, &framebufferCreateInfo, nullptr, &vk.Framebuffers[i]));
  }
}

inline U32 FindMemoryType(U32 typeFilter, VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties& memory = vk.PhysicalDeviceInfo.Memory;
  for (U32 i = 0; i < memory.memoryTypeCount; i++) {
    if (typeFilter & (1 << i) && (memory.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }
  throw std::runtime_error("Failed to find suitable memory type!");
}

inline VkFormat FindImageFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling,
                                VkFormatFeatureFlags features) {
  for (VkFormat format : candidates) {
    VkFormatProperties props;
    vkGetPhysicalDeviceFormatProperties(vk.PhysicalDevice, format, &props);
    if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
      return format;
    } else if (tiling == VK_IMAGE_TILING_OPTIMAL &&
               (props.optimalTilingFeatures & features) == features) {
      return format;
    }
  }
  throw std::runtime_error("Failed to find suitable depth format!");
}

bool CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                  VkBuffer& buffer, VkDeviceMemory& deviceMemory, U64* bufferAlignment) {
  const VkBufferCreateInfo bufferCreateInfo{
      VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,  // sType
      nullptr,                               // pNext
      0,                                     // flags
      size,                                  // size
      usage,                                 // usage
      VK_SHARING_MODE_EXCLUSIVE,             // sharingMode
      0,                                     // queueFamilyIndexCount
      nullptr                                // pQueueFamilyIndices
  };
  VkCall(vkCreateBuffer(vk.Device, &bufferCreateInfo, nullptr, &buffer));

  VkMemoryRequirements memoryRequirements;
  vkGetBufferMemoryRequirements(vk.Device, buffer, &memoryRequirements);
  if (bufferAlignment != nullptr) {
    *bufferAlignment = memoryRequirements.alignment;
  }

  const VkMemoryAllocateInfo allocateInfo{
      VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,                        // sType
      nullptr,                                                       // pNext
      memoryRequirements.size,                                       // allocationSize
      FindMemoryType(memoryRequirements.memoryTypeBits, properties)  // memoryTypeIndex
  };

  VkCall(vkAllocateMemory(vk.Device, &allocateInfo, nullptr, &deviceMemory));
  vkBindBufferMemory(vk.Device, buffer, deviceMemory, 0);

  return true;
}

void CopyBuffer(VkBuffer source, VkBuffer destination, VkDeviceSize size) {
  const VkCommandBufferAllocateInfo allocateInfo{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,  // sType
      nullptr,                                         // pNext
      vk.TransferCommandPool,                          // commandPool
      VK_COMMAND_BUFFER_LEVEL_PRIMARY,                 // level
      1                                                // commandBufferCount
  };

  VkCommandBuffer cmdBuf;
  vkAllocateCommandBuffers(vk.Device, &allocateInfo, &cmdBuf);

  constexpr static const VkCommandBufferBeginInfo beginInfo{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,  // sType
      nullptr,                                      // pNext
      VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,  // flags
      nullptr                                       // pInheritanceInfo
  };
  VkCall(vkBeginCommandBuffer(cmdBuf, &beginInfo));

  const VkBufferCopy copyRegion{
      0,    // srcOffset
      0,    // dstOffset
      size  // size
  };
  vkCmdCopyBuffer(cmdBuf, source, destination, 1, &copyRegion);

  VkCall(vkEndCommandBuffer(cmdBuf));

  VkSubmitInfo submitInfo{
      VK_STRUCTURE_TYPE_SUBMIT_INFO,  // sType
      nullptr,                        // pNext
      0,                              // waitSemaphoreCount
      nullptr,                        // pWaitSemaphores
      nullptr,                        // pWaitDstStageMask
      1,                              // commandBufferCount
      &cmdBuf,                        // pCommandBuffers
      0,                              // signalSemaphoreCount
      nullptr                         // pSignalSemaphores
  };

  vkQueueSubmit(vk.TransferQueue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(vk.TransferQueue);
  vkFreeCommandBuffers(vk.Device, vk.TransferCommandPool, 1, &cmdBuf);
}

void CopyBufferToImage(VkBuffer source, VkImage destination, U32 width, U32 height) {
  const VkCommandBufferAllocateInfo allocateInfo{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,  // sType
      nullptr,                                         // pNext
      vk.TransferCommandPool,                          // commandPool
      VK_COMMAND_BUFFER_LEVEL_PRIMARY,                 // level
      1                                                // commandBufferCount
  };

  VkCommandBuffer cmdBuf;
  vkAllocateCommandBuffers(vk.Device, &allocateInfo, &cmdBuf);

  constexpr static const VkCommandBufferBeginInfo beginInfo{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,  // sType
      nullptr,                                      // pNext
      VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,  // flags
      nullptr                                       // pInheritanceInfo
  };
  VkCall(vkBeginCommandBuffer(cmdBuf, &beginInfo));
  VkBufferImageCopy copyRegion{};
  copyRegion.bufferOffset = 0;
  copyRegion.bufferRowLength = 0;
  copyRegion.bufferImageHeight = 0;
  copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  copyRegion.imageSubresource.mipLevel = 0;
  copyRegion.imageSubresource.baseArrayLayer = 0;
  copyRegion.imageSubresource.layerCount = 1;
  copyRegion.imageOffset = {0, 0, 0};
  copyRegion.imageExtent = {width, height, 1};
  vkCmdCopyBufferToImage(cmdBuf, source, destination, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                         &copyRegion);
  VkCall(vkEndCommandBuffer(cmdBuf));

  VkSubmitInfo submitInfo{
      VK_STRUCTURE_TYPE_SUBMIT_INFO,  // sType
      nullptr,                        // pNext
      0,                              // waitSemaphoreCount
      nullptr,                        // pWaitSemaphores
      nullptr,                        // pWaitDstStageMask
      1,                              // commandBufferCount
      &cmdBuf,                        // pCommandBuffers
      0,                              // signalSemaphoreCount
      nullptr                         // pSignalSemaphores
  };

  vkQueueSubmit(vk.TransferQueue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(vk.TransferQueue);
  vkFreeCommandBuffers(vk.Device, vk.TransferCommandPool, 1, &cmdBuf);
}

bool UploadMesh(Mesh& mesh) {
  const VkDeviceSize vertexSize = sizeof(mesh.Vertices[0]) * mesh.Vertices.size();
  const VkDeviceSize indexSize = sizeof(mesh.Indices[0]) * mesh.Indices.size();
  const VkDeviceSize bufferSize = vertexSize + indexSize;
  mesh.IndicesOffset = vertexSize;

  VkBuffer stagingBuffer;
  VkDeviceMemory stagingDeviceMemory;
  ASSERT(CreateBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      stagingBuffer, stagingDeviceMemory));
  void* data;
  vkMapMemory(vk.Device, stagingDeviceMemory, 0, bufferSize, 0, &data);
  memcpy(data, mesh.Vertices.data(), static_cast<size_t>(vertexSize));
  memcpy(static_cast<U8*>(data) + vertexSize, mesh.Indices.data(), static_cast<size_t>(indexSize));
  vkUnmapMemory(vk.Device, stagingDeviceMemory);

  ASSERT(CreateBuffer(bufferSize,
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                          VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mesh.Buffer, mesh.DeviceMemory));

  CopyBuffer(stagingBuffer, mesh.Buffer, bufferSize);

  vkDestroyBuffer(vk.Device, stagingBuffer, nullptr);
  vkFreeMemory(vk.Device, stagingDeviceMemory, nullptr);

  return true;
}

void FreeMesh(const Mesh& mesh) {
  vkDestroyBuffer(vk.Device, mesh.Buffer, nullptr);
  vkFreeMemory(vk.Device, mesh.DeviceMemory, nullptr);
}

void Run() {
  // =====
  // Create our window
  // =====
  app.Instance = GetModuleHandle(NULL);
  app.Cursor = LoadCursor(NULL, IDC_ARROW);

  // Register window class
  {
    static const WNDCLASSW wc{
        0,                           // style
        ApplicationWindowProcedure,  // lpfnWndProc
        0,                           // cbClsExtra
        0,                           // cbWndExtra
        app.Instance,                // hInstance
        NULL,                        // hIcon
        app.Cursor,                  // hCursor
        NULL,                        // hbrBackground
        NULL,                        // lpszMenuName
        config.WindowClassName       // lpszClassName
    };
    RegisterClassW(&wc);
  }

  // Create application window
  {
    constexpr U32 windowStyle =
        WS_OVERLAPPED | WS_SYSMENU | WS_CAPTION | WS_THICKFRAME | WS_MAXIMIZEBOX | WS_MINIMIZEBOX;
    constexpr U32 windowExStyle = WS_EX_APPWINDOW;

    U32 windowW = config.WindowWidth;
    U32 windowH = config.WindowHeight;
    U32 windowX = 100;
    U32 windowY = 100;

    RECT borderRect = {0, 0, 0, 0};
    AdjustWindowRectEx(&borderRect, windowStyle, false, windowExStyle);
    windowX += borderRect.left;
    windowY += borderRect.top;
    windowW += borderRect.right - borderRect.left;
    windowH += borderRect.bottom - borderRect.top;

    LogTrace("Creating %dx%d application window. (Actual size: %dx%d)", config.WindowWidth,
             config.WindowHeight, windowW, windowH);

    app.Window = CreateWindowExW(windowExStyle, config.WindowClassName, config.WindowTitle,
                                 windowStyle, windowX, windowY, windowW, windowH, nullptr, nullptr,
                                 app.Instance, nullptr);

    if (!app.Window) {
      throw std::runtime_error("Failed to create window!");
    }

    ShowWindow(app.Window, SW_SHOW);
  }

  // =====
  // Vulkan Instance
  // =====
  constexpr static const VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{
      VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,  // sType
      nullptr,                                                  // pNext
      0,                                                        // flags
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
          VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,  // messageSeverity
      VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
          VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT |
          VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT,  // messageType
      VulkanDebugCallback,                                 // pfnUserCallback
      nullptr                                              // pUserData
  };
  {
    constexpr static VkApplicationInfo appInfo{
        VK_STRUCTURE_TYPE_APPLICATION_INFO,  // sType
        nullptr,                             // pNext
        "Vulkan",                            // pApplicationName
        VK_MAKE_VERSION(1, 0, 0),            // applicationVersion
        "No Engine",                         // pEngineName
        VK_MAKE_VERSION(1, 0, 0),            // engineVersion
        VK_API_VERSION_1_1                   // apiVersion
    };

#ifdef DEBUG
    constexpr static const char* instanceExtensions[] = {"VK_KHR_surface", "VK_KHR_win32_surface",
                                                         VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
    constexpr static const char* instanceLayers[] = {"VK_LAYER_KHRONOS_validation"};
    constexpr static const void* pNext = static_cast<const void*>(&debugCreateInfo);
#else
    constexpr static const char* instanceExtensions[] = {"VK_KHR_surface", "VK_KHR_win32_surface"};
    constexpr static const char** instanceLayers = nullptr;
    constexpr static const void* pNext = nullptr;
#endif

    constexpr static U32 instanceCount = sizeof(instanceExtensions) / sizeof(*instanceExtensions);
    constexpr static U32 layerCount = sizeof(instanceLayers) / sizeof(*instanceLayers);

    constexpr static VkInstanceCreateInfo createInfo{
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,  // sType
        pNext,                                   // pNext
        0,                                       // flags
        &appInfo,                                // pApplicationInfo
        layerCount,                              // enabledLayerCount
        instanceLayers,                          // ppEnabledLayerNames
        instanceCount,                           // enabledExtensionCount
        instanceExtensions                       // ppEnabledExtensionNames
    };

    // Verify required extensions are present
    {
      U32 availableExtensionCount = 0;
      VkCall(vkEnumerateInstanceExtensionProperties(nullptr, &availableExtensionCount, nullptr));
      std::vector<VkExtensionProperties> availableExtensions(availableExtensionCount);
      VkCall(vkEnumerateInstanceExtensionProperties(nullptr, &availableExtensionCount,
                                                    availableExtensions.data()));

      for (U32 i = 0; i < instanceCount; i++) {
        bool found = false;
        for (U32 j = 0; j < availableExtensionCount; j++) {
          if (strcmp(instanceExtensions[i], availableExtensions[j].extensionName) == 0) {
            found = true;
            break;
          }
        }
        if (!found) {
          throw std::runtime_error(std::string("Failed to load Vulkan instance extension: ") +
                                   instanceExtensions[i]);
        }
      }
    }

    // Verify required layers are present
    {
      U32 availableLayerCount = 0;
      VkCall(vkEnumerateInstanceLayerProperties(&availableLayerCount, nullptr));
      std::vector<VkLayerProperties> availableLayers(availableLayerCount);
      VkCall(vkEnumerateInstanceLayerProperties(&availableLayerCount, availableLayers.data()));

      for (U32 i = 0; i < layerCount; i++) {
        bool found = false;
        for (U32 j = 0; j < availableLayerCount; j++) {
          if (strcmp(instanceLayers[i], availableLayers[j].layerName) == 0) {
            found = true;
            break;
          }
        }
        if (!found) {
          throw std::runtime_error(std::string("Failed to load Vulkan instance layer: ") +
                                   instanceLayers[i]);
        }
      }
    }

    VkCall(vkCreateInstance(&createInfo, nullptr, &vk.Instance));
  }

  // =====
  // Vulkan Debug Messenger
  // =====
  {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        vk.Instance, "vkCreateDebugUtilsMessengerEXT");
    if (func) {
      VkCall(func(vk.Instance, &debugCreateInfo, nullptr, &vk.DebugMessenger));
    } else {
      throw std::runtime_error("Failed to locate function vkCreateDebugUtilsMessengerEXT!");
    }
  }

  // =====
  // Vulkan Surface
  // =====
  {
    const static VkWin32SurfaceCreateInfoKHR createInfo{
        VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,  // sType
        nullptr,                                          // pNext
        0,                                                // flags
        app.Instance,                                     // hinstance
        app.Window                                        // hwnd
    };

    VkCall(vkCreateWin32SurfaceKHR(vk.Instance, &createInfo, nullptr, &vk.Surface));
  }

  // =====
  // Vulkan Physical Device
  // =====
  {
    U32 deviceCount = 0;
    VkCall(vkEnumeratePhysicalDevices(vk.Instance, &deviceCount, nullptr));
    if (deviceCount == 0) {
      throw std::runtime_error("No Vulkan devices found!");
    }
    LogTrace("Found %d Vulkan devices.", deviceCount);
    std::vector<VkPhysicalDevice> devices(deviceCount);
    std::vector<VulkanPhysicalDeviceInfo> deviceInfos(deviceCount);
    VkCall(vkEnumeratePhysicalDevices(vk.Instance, &deviceCount, devices.data()));

    for (U32 deviceIndex = 0; deviceIndex < deviceCount; deviceIndex++) {
      const VkPhysicalDevice& device = devices[deviceIndex];
      VulkanPhysicalDeviceInfo& info = deviceInfos[deviceIndex];

      info.Device = device;
      vkGetPhysicalDeviceFeatures(device, &info.Features);
      vkGetPhysicalDeviceMemoryProperties(device, &info.Memory);
      vkGetPhysicalDeviceProperties(device, &info.Properties);

      // Enumerate device queues
      {
        info.Queues.GraphicsIndex = -1;
        info.Queues.TransferIndex = -1;
        info.Queues.ComputeIndex = -1;
        info.Queues.PresentationIndex = -1;

        vkGetPhysicalDeviceQueueFamilyProperties(device, &info.Queues.Count, nullptr);
        std::vector<VkQueueFamilyProperties> families(info.Queues.Count);
        info.Queues.Queues.resize(info.Queues.Count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &info.Queues.Count, families.data());

        for (U32 i = 0; i < info.Queues.Count; i++) {
          VulkanPhysicalDeviceQueue& queue = info.Queues.Queues[i];
          queue.Index = i;
          queue.Flags = families[i].queueFlags;
          queue.Count = families[i].queueCount;
          vkGetPhysicalDeviceSurfaceSupportKHR(device, i, vk.Surface, &queue.PresentKHR);

          if (info.Queues.GraphicsIndex == -1 && queue.SupportsGraphics()) {
            info.Queues.GraphicsIndex = i;
          }
          if (info.Queues.PresentationIndex == -1 && queue.SupportsPresentation()) {
            info.Queues.PresentationIndex = i;
          }

          if (queue.SupportsCompute() && (info.Queues.ComputeIndex == -1 ||
                                          info.Queues.ComputeIndex == info.Queues.GraphicsIndex)) {
            info.Queues.ComputeIndex = i;
          }
          if (queue.SupportsTransfer() &&
              (info.Queues.TransferIndex == -1 ||
               info.Queues.TransferIndex == info.Queues.GraphicsIndex)) {
            info.Queues.TransferIndex = i;
          }
        }
      }

      // Query device swapchain support
      {
        VkCall(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, vk.Surface,
                                                         &info.SwapchainSupport.Capabilities));

        U32 formatCount = 0;
        VkCall(vkGetPhysicalDeviceSurfaceFormatsKHR(device, vk.Surface, &formatCount, nullptr));
        if (formatCount != 0) {
          info.SwapchainSupport.Formats.resize(formatCount);
          VkCall(vkGetPhysicalDeviceSurfaceFormatsKHR(device, vk.Surface, &formatCount,
                                                      info.SwapchainSupport.Formats.data()));
        }

        U32 presentModeCount = 0;
        VkCall(vkGetPhysicalDeviceSurfacePresentModesKHR(device, vk.Surface, &presentModeCount,
                                                         nullptr));
        if (presentModeCount != 0) {
          info.SwapchainSupport.PresentationModes.resize(presentModeCount);
          VkCall(vkGetPhysicalDeviceSurfacePresentModesKHR(
              device, vk.Surface, &presentModeCount,
              info.SwapchainSupport.PresentationModes.data()));
        }
      }

      // Query device extensions
      {
        U32 extensionCount = 0;
        VkCall(vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr));
        if (extensionCount != 0) {
          info.Extensions.resize(extensionCount);
          VkCall(vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                                      info.Extensions.data()));
        }
      }

#ifdef TRACE
      // Dump physical device info for debugging
      {
        LogTrace("Vulkan Device \"%s\":", info.Properties.deviceName);

        // General device details
        LogTrace(" - Vulkan API: %d.%d.%d", VK_VERSION_MAJOR(info.Properties.apiVersion),
                 VK_VERSION_MINOR(info.Properties.apiVersion),
                 VK_VERSION_PATCH(info.Properties.apiVersion));
        switch (info.Properties.deviceType) {
          case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
            LogTrace(" - Device Type: Dedicated");
            break;
          case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
            LogTrace(" - Device Type: Integrated");
            break;
          case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
            LogTrace(" - Device Type: Virtual");
            break;
          case VK_PHYSICAL_DEVICE_TYPE_CPU:
            LogTrace(" - Device Type: CPU");
            break;
          default:
            LogTrace(" - Device Type: Unknown");
            break;
        }
        LogTrace(" - Max 2D Resolution: %d", info.Properties.limits.maxImageDimension2D);

        // Memory Details
        LogTrace(" - Memory:");
        LogTrace("   - Types (%d):", info.Memory.memoryTypeCount);
        // DL = Device Local
        // HV = Host Visible
        // HC = Host Coherent
        // HH = Host Cached
        // LA = Lazily Allocated
        // PT = Protected
        // DC = Device Coherent (AMD)
        // DU = Device Uncached (AMD)
        LogTrace("               / DL | HV | HC | HH | LA | PT | DC | DU \\");
        for (U32 memoryTypeIndex = 0; memoryTypeIndex < info.Memory.memoryTypeCount;
             memoryTypeIndex++) {
          const VkMemoryType& memType = info.Memory.memoryTypes[memoryTypeIndex];
          LogTrace(
              "     - Heap %d: | %s | %s | %s | %s | %s | %s | %s | %s |", memType.heapIndex,
              memType.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT ? "DL" : "  ",
              memType.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT ? "HV" : "  ",
              memType.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT ? "HC" : "  ",
              memType.propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT ? "HH" : "  ",
              memType.propertyFlags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT ? "LA" : "  ",
              memType.propertyFlags & VK_MEMORY_PROPERTY_PROTECTED_BIT ? "PT" : "  ",
              memType.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD ? "DC" : "  ",
              memType.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD ? "DU" : "  ");
        }
        LogTrace("   - Heaps (%d):", info.Memory.memoryHeapCount);
        for (U32 memoryHeapIndex = 0; memoryHeapIndex < info.Memory.memoryHeapCount;
             memoryHeapIndex++) {
          const VkMemoryHeap& memHeap = info.Memory.memoryHeaps[memoryHeapIndex];
          // DL = Device Local
          // MI = Multi Instance
          // MI = Multi Instance (KHR)
          LogTrace("     - Heap %d: %.2f MiB { %s | %s | %s }", memoryHeapIndex,
                   ((F32)memHeap.size / 1024.0f / 1024.0f),
                   memHeap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT ? "DL" : "  ",
                   memHeap.flags & VK_MEMORY_HEAP_MULTI_INSTANCE_BIT ? "MI" : "  ",
                   memHeap.flags & VK_MEMORY_HEAP_MULTI_INSTANCE_BIT_KHR ? "MK" : "  ");
        }

        LogTrace(" - Queue Families (%d):", info.Queues.Count);
        // GFX = Graphics
        // CMP = Compute
        // TRA = Transfer
        // SPB = Sparse Binding
        // PRT = Protected
        // PST = Presentation (KHR)
        LogTrace("             / GFX  | CMP  | TRA  | SPB  | PRT  | PST  \\");
        const VulkanPhysicalDeviceQueues& queues = info.Queues;
        for (U32 queueIndex = 0; queueIndex < info.Queues.Count; queueIndex++) {
          const VulkanPhysicalDeviceQueue& queue = info.Queues.Queues[queueIndex];
          // Asterisk indicates the Queue Family has been selected for that particular queue
          // operation.
          LogTrace(" - Family %d: { %s%c | %s%c | %s%c | %s%c | %s%c | %s%c } (%d Queues)",
                   queueIndex, queue.SupportsGraphics() ? "GFX" : "   ",
                   queues.GraphicsIndex == queue.Index ? '*' : ' ',
                   queue.SupportsCompute() ? "CMP" : "   ",
                   queues.ComputeIndex == queue.Index ? '*' : ' ',
                   queue.SupportsTransfer() ? "TRA" : "   ",
                   queues.TransferIndex == queue.Index ? '*' : ' ',
                   queue.SupportsSparseBinding() ? "SPB" : "   ", ' ',
                   queue.SupportsProtected() ? "PRT" : "   ", ' ',
                   queue.SupportsPresentation() ? "PST" : "   ",
                   queues.PresentationIndex == queue.Index ? '*' : ' ', queue.Count);
        }

        // Swapchain details
        LogTrace("-- Swapchain:");
        LogTrace("---- Image Count: %d Min / %d Max",
                 info.SwapchainSupport.Capabilities.minImageCount,
                 info.SwapchainSupport.Capabilities.maxImageCount);
        LogTrace("---- Image Size: %dx%d Min / %dx%d Max",
                 info.SwapchainSupport.Capabilities.minImageExtent.width,
                 info.SwapchainSupport.Capabilities.minImageExtent.height,
                 info.SwapchainSupport.Capabilities.maxImageExtent.width,
                 info.SwapchainSupport.Capabilities.maxImageExtent.height);
        LogTrace("---- Image Formats: %d", info.SwapchainSupport.Formats.size());
        LogTrace("---- Present Modes: %d", info.SwapchainSupport.PresentationModes.size());

        // Extensions
        LogTrace("-- Extensions (%d):", info.Extensions.size());
        for (const VkExtensionProperties& ext : info.Extensions) {
          LogTrace("---- %s", ext.extensionName);
        }
      }
#endif
    }

    for (U32 deviceIndex = 0; deviceIndex < deviceCount; deviceIndex++) {
      VulkanPhysicalDeviceInfo& info = deviceInfos[deviceIndex];
      LogTrace("Considering device: %s", info.Properties.deviceName);

      if (info.Queues.GraphicsIndex == -1 || info.Queues.PresentationIndex == -1) {
        LogTrace("Rejecting device: Missing graphics or presentation queue.");
        continue;
      }

      constexpr static const char* deviceExtensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
      constexpr static U32 extensionCount = sizeof(deviceExtensions) / sizeof(*deviceExtensions);

      bool success = true;
      for (U32 i = 0; i < extensionCount; i++) {
        bool found = false;
        for (U32 j = 0; j < info.Extensions.size(); j++) {
          if (strcmp(deviceExtensions[i], info.Extensions[j].extensionName) == 0) {
            found = true;
            break;
          }
        }
        if (!found) {
          LogTrace("Rejecting device: Failed to find required extension \"%s\"",
                   deviceExtensions[i]);
          success = false;
          break;
        }
      }
      if (!success) {
        continue;
      }

      if (info.SwapchainSupport.Formats.size() == 0 ||
          info.SwapchainSupport.PresentationModes.size() == 0) {
        LogTrace("Rejecting device: No available image formats or presentation modes.");
        continue;
      }

      if (info.Features.samplerAnisotropy == VK_FALSE) {
        LogTrace("Rejecting device: No support for sampler anisotropy.");
        continue;
      }

      vk.PhysicalDevice = info.Device;
      vk.PhysicalDeviceInfo = info;
    }

    if (vk.PhysicalDevice == VK_NULL_HANDLE) {
      throw std::runtime_error("No available Vulkan devices meet requirements!");
    } else {
      LogDebug("Using physical device: %s", vk.PhysicalDeviceInfo.Properties.deviceName);
    }
  }

  // =====
  // Vulkan Logical Device
  // =====
  {
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos(vk.PhysicalDeviceInfo.Queues.Count);
    constexpr static const float queuePriority = 1.0f;
    for (U32 i = 0; i < vk.PhysicalDeviceInfo.Queues.Count; i++) {
      queueCreateInfos[i].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      queueCreateInfos[i].queueFamilyIndex = vk.PhysicalDeviceInfo.Queues.Queues[i].Index;
      queueCreateInfos[i].queueCount = 1;
      queueCreateInfos[i].pQueuePriorities = &queuePriority;
    }

    static VkPhysicalDeviceFeatures requestedFeatures{};
    requestedFeatures.samplerAnisotropy = VK_TRUE;

    constexpr static const char* deviceExtensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
#ifdef DEBUG
    constexpr static const char* deviceLayers[] = {"VK_LAYER_KHRONOS_validation"};
#else
    constexpr static const char** deviceLayers = nullptr;
#endif

    constexpr static const U32 extensionCount =
        sizeof(deviceExtensions) / sizeof(*deviceExtensions);
    constexpr static const U32 layerCount = sizeof(deviceLayers) / sizeof(*deviceLayers);

    static const VkDeviceCreateInfo createInfo{
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,  // sType
        nullptr,                               // pNext
        0,                                     // flags
        vk.PhysicalDeviceInfo.Queues.Count,    // queueCreateInfoCount
        queueCreateInfos.data(),               // pQueueCreateInfos
        layerCount,                            // enabledLayerCount
        deviceLayers,                          // ppEnabledLayerNames
        extensionCount,                        // enabledExtensionCount
        deviceExtensions,                      // ppEnabledExtensionNames
        &requestedFeatures                     // pEnabledFeatures
    };

    VkCall(vkCreateDevice(vk.PhysicalDevice, &createInfo, nullptr, &vk.Device));

    vkGetDeviceQueue(vk.Device, vk.PhysicalDeviceInfo.Queues.GraphicsIndex, 0, &vk.GraphicsQueue);
    vkGetDeviceQueue(vk.Device, vk.PhysicalDeviceInfo.Queues.TransferIndex, 0, &vk.TransferQueue);
    vkGetDeviceQueue(vk.Device, vk.PhysicalDeviceInfo.Queues.PresentationIndex, 0,
                     &vk.PresentationQueue);
  }

  // =====
  // Vulkan Command Pools
  // =====
  {
    VkCommandPoolCreateInfo graphicsPoolCreateInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    graphicsPoolCreateInfo.queueFamilyIndex = vk.PhysicalDeviceInfo.Queues.GraphicsIndex;
    graphicsPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VkCall(
        vkCreateCommandPool(vk.Device, &graphicsPoolCreateInfo, nullptr, &vk.GraphicsCommandPool));

    VkCommandPoolCreateInfo transferPoolCreateInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    transferPoolCreateInfo.queueFamilyIndex = vk.PhysicalDeviceInfo.Queues.TransferIndex;
    transferPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VkCall(
        vkCreateCommandPool(vk.Device, &transferPoolCreateInfo, nullptr, &vk.TransferCommandPool));
  }

  // =====
  // Determine Swapchain format
  // =====
  {
    // Choose swapchain format
    for (const auto& availableFormat : vk.PhysicalDeviceInfo.SwapchainSupport.Formats) {
      if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
          availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
        vk.SurfaceFormat = availableFormat;
      }
    }
    vk.SurfaceFormat = vk.PhysicalDeviceInfo.SwapchainSupport.Formats[0];

    // Choose presentation mode
    for (const auto& presentMode : vk.PhysicalDeviceInfo.SwapchainSupport.PresentationModes) {
      if (presentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
        vk.PresentMode = presentMode;
      }
    }
    vk.PresentMode = VK_PRESENT_MODE_FIFO_KHR;

    vk.DepthFormat = FindImageFormat(
        {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
        VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
  }

  // =====
  // Vulkan Render Pass
  // =====
  {
    const VkAttachmentDescription colorAttachment{
        0,                                 // flags
        vk.SurfaceFormat.format,           // format
        VK_SAMPLE_COUNT_1_BIT,             // samples
        VK_ATTACHMENT_LOAD_OP_CLEAR,       // loadOp
        VK_ATTACHMENT_STORE_OP_STORE,      // storeOp
        VK_ATTACHMENT_LOAD_OP_DONT_CARE,   // stencilLoadOp
        VK_ATTACHMENT_STORE_OP_DONT_CARE,  // stencilStoreOp
        VK_IMAGE_LAYOUT_UNDEFINED,         // initialLayout
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR    // finalLayout
    };

    const VkAttachmentDescription depthAttachment{
        0,                                                // flags
        vk.DepthFormat,                                   // format
        VK_SAMPLE_COUNT_1_BIT,                            // samples
        VK_ATTACHMENT_LOAD_OP_CLEAR,                      // loadOp
        VK_ATTACHMENT_STORE_OP_DONT_CARE,                 // storeOp
        VK_ATTACHMENT_LOAD_OP_DONT_CARE,                  // stencilLoadOp
        VK_ATTACHMENT_STORE_OP_DONT_CARE,                 // stencilStoreOp
        VK_IMAGE_LAYOUT_UNDEFINED,                        // initialLayout
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL  // finalLayout
    };

    constexpr static VkAttachmentReference colorAttachmentRef{
        0,                                        // attachment
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL  // layout
    };

    constexpr static VkAttachmentReference depthAttachmentRef{
        1,                                                // attachment
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL  // layout
    };

    constexpr static VkSubpassDescription subpass{
        0,                                // flags
        VK_PIPELINE_BIND_POINT_GRAPHICS,  // pipelineBindPoint
        0,                                // inputAttachmentCount
        nullptr,                          // pInputAttachments
        1,                                // colorAttachmentCount
        &colorAttachmentRef,              // pColorAttachments
        nullptr,                          // pResolveAttachments
        &depthAttachmentRef,              // pDepthStencilAttachment
        0,                                // preserveAttachmentCount
        nullptr,                          // pPreserveAttachments
    };

    constexpr static VkSubpassDependency dependency{
        VK_SUBPASS_EXTERNAL,                            // srcSubpass
        0,                                              // dstSubpass
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,  // srcStageMask
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,  // dstStageMask
        0,                                              // srcAccessMask
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,           // dstAccessMask
        0                                               // dependencyFlags
    };

    const static VkAttachmentDescription attachments[] = {colorAttachment, depthAttachment};

    const static VkRenderPassCreateInfo renderPassCreateInfo{
        VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,  // sType
        nullptr,                                    // pNext
        0,                                          // flags
        ArrLen(attachments),                        // attachmentCount
        attachments,                                // pAttachments
        1,                                          // subpassCount
        &subpass,                                   // pSubpasses
        1,                                          // dependencyCount
        &dependency                                 // pDependencies
    };
    VkCall(vkCreateRenderPass(vk.Device, &renderPassCreateInfo, nullptr, &vk.RenderPass));
  }

  // =====
  // Vulkan Swapchain
  // =====
  CreateSwapchain();

  // =====
  // Vulkan Command Buffers
  // =====
  {
    vk.CommandBuffers.resize(vk.SwapchainImageCount);

    const VkCommandBufferAllocateInfo allocateInfo{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,  // sType
        nullptr,                                         // pNext
        vk.GraphicsCommandPool,                          // commandPool
        VK_COMMAND_BUFFER_LEVEL_PRIMARY,                 // level
        vk.SwapchainImageCount                           // commandBufferCount
    };

    VkCall(vkAllocateCommandBuffers(vk.Device, &allocateInfo, vk.CommandBuffers.data()));
  }

  // =====
  // Vulkan Sync Objects
  // =====
  {
    vk.ImageAvailableSemaphores.resize(config.MaxImagesInFlight);
    vk.RenderFinishedSemaphores.resize(config.MaxImagesInFlight);
    vk.InFlightFences.resize(config.MaxImagesInFlight);
    vk.ImagesInFlight.resize(vk.SwapchainImageCount);

    constexpr static const VkSemaphoreCreateInfo semaphoreCreateInfo{
        VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,  // sType
        nullptr,                                  // pNext
        0                                         // flags
    };

    constexpr static const VkFenceCreateInfo fenceCreateInfo{
        VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,  // sType
        nullptr,                              // pNext
        VK_FENCE_CREATE_SIGNALED_BIT          // flags
    };

    for (U32 i = 0; i < config.MaxImagesInFlight; i++) {
      VkCall(vkCreateSemaphore(vk.Device, &semaphoreCreateInfo, nullptr,
                               &vk.ImageAvailableSemaphores[i]));
      VkCall(vkCreateSemaphore(vk.Device, &semaphoreCreateInfo, nullptr,
                               &vk.RenderFinishedSemaphores[i]));
      VkCall(vkCreateFence(vk.Device, &fenceCreateInfo, nullptr, &vk.InFlightFences[i]));
    }
  }

  // =====
  // Load assets
  // =====
  Object* obj = new Object(new Mesh("assets/models/viking_room.obj"),
                           new UnlitTexturedMaterial("assets/textures/viking_room.png"));
  //obj->Enable();
  Object* cube = new Object(new Mesh("assets/models/cube.obj"), new LitMaterial());
  cube->Enable();
  Object* lightCube = new Object(new Mesh("assets/models/cube.obj"), new UnlitMaterial());
  lightCube->Enable();

  // =====
  // Main application loop
  // =====
  LogTrace("Beginning main application loop.");
  U32 currentFrame = 0;
  while (!app.CloseRequested) {
    MSG message;
    while (PeekMessageW(&message, NULL, 0, 0, PM_REMOVE)) {
      TranslateMessage(&message);
      DispatchMessageW(&message);
    }

    // Render
    {
      U32 imageIndex;

      vkWaitForFences(vk.Device, 1, &vk.InFlightFences[currentFrame], VK_TRUE, U64_MAX);

      // Acquire framebuffer
      {
        VkResult acquireImageResult = vkAcquireNextImageKHR(
            vk.Device, vk.Swapchain, U64_MAX, vk.ImageAvailableSemaphores[currentFrame],
            VK_NULL_HANDLE, &imageIndex);
        if (acquireImageResult == VK_ERROR_OUT_OF_DATE_KHR) {
          CreateSwapchain();
        } else if (acquireImageResult != VK_SUCCESS && acquireImageResult != VK_SUBOPTIMAL_KHR) {
          throw std::runtime_error("Failed to acquire image!");
        }

        if (vk.ImagesInFlight[imageIndex] != VK_NULL_HANDLE) {
          vkWaitForFences(vk.Device, 1, &vk.ImagesInFlight[imageIndex], VK_TRUE, U64_MAX);
        }
        vk.ImagesInFlight[imageIndex] = vk.InFlightFences[currentFrame];
      }

      // Record command buffer
      VkCommandBuffer& cmdBuf = vk.CommandBuffers[imageIndex];
      {
        // vkBeginCommandBuffer
        {
          VkCommandBufferBeginInfo beginInfo{
              VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,  // sType
              nullptr,                                      // pNext
              0,                                            // flags
              nullptr                                       // pInheritanceInfo
          };
          VkCall(vkBeginCommandBuffer(cmdBuf, &beginInfo));
        }

        // vkCmdBeginRenderPass
        {
          VkClearValue clearValues[] = {{0.05f, 0.05f, 0.05f, 1.0f}, {1.0f, 0}};
          VkRenderPassBeginInfo beginInfo{
              VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,  // sType
              nullptr,                                   // pNext
              vk.RenderPass,                             // renderPass
              vk.Framebuffers[imageIndex],               // framebuffer
              {
                  {0, 0},             // renderArea.offset
                  vk.SwapchainExtent  // renderArea.extent
              },                      // renderArea
              ArrLen(clearValues),    // clearValueCount
              clearValues             // pClearValues
          };
          vkCmdBeginRenderPass(cmdBuf, &beginInfo, VK_SUBPASS_CONTENTS_INLINE);
        }

        for (const auto& object : vk.ObjectsToRender) {
          object->ObjectMaterial->Update(imageIndex);
          object->ObjectMaterial->Bind(cmdBuf, imageIndex);

          // Bind meshes
          static const U64 offsets = 0;
          vkCmdBindVertexBuffers(cmdBuf, 0, 1, &object->ObjectMesh->Buffer, &offsets);
          vkCmdBindIndexBuffer(cmdBuf, object->ObjectMesh->Buffer,
                               object->ObjectMesh->IndicesOffset, VK_INDEX_TYPE_UINT32);

          // Draw mesh
          vkCmdDrawIndexed(cmdBuf, static_cast<U32>(object->ObjectMesh->Indices.size()), 1, 0, 0,
                           0);
        }

        // vkCmdEndRenderPass
        { vkCmdEndRenderPass(cmdBuf); }

        // vkEndCommandBuffer
        { VkCall(vkEndCommandBuffer(cmdBuf)); }
      }

      vkResetFences(vk.Device, 1, &vk.InFlightFences[currentFrame]);

      const VkSemaphore waitSemaphores[] = {vk.ImageAvailableSemaphores[currentFrame]};
      const VkSemaphore signalSemaphores[] = {vk.RenderFinishedSemaphores[currentFrame]};

      // Submit command buffer
      {
        constexpr static const VkPipelineStageFlags waitStages[] = {
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        const VkSubmitInfo submitInfo{
            VK_STRUCTURE_TYPE_SUBMIT_INFO,                         // sType
            nullptr,                                               // pNext
            sizeof(waitSemaphores) / sizeof(*waitSemaphores),      // waitSemaphoreCount
            waitSemaphores,                                        // pWaitSemaphores
            waitStages,                                            // pWaitDstStageMask
            1,                                                     // commandBufferCount
            &cmdBuf,                                               // pCommandBuffers
            sizeof(signalSemaphores) / sizeof(*signalSemaphores),  // signalSemaphoreCount
            signalSemaphores                                       // pSignalSemaphores
        };

        VkCall(vkQueueSubmit(vk.GraphicsQueue, 1, &submitInfo, vk.InFlightFences[currentFrame]));
      }

      // Present swapchain image
      {
        const VkSwapchainKHR swapchains[] = {vk.Swapchain};
        VkPresentInfoKHR presentInfo{
            VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,                    // sType
            nullptr,                                               // pNext
            sizeof(signalSemaphores) / sizeof(*signalSemaphores),  // waitSemaphoreCount
            signalSemaphores,                                      // pWaitSemaphores
            1,                                                     // swapchainCount
            swapchains,                                            // pSwapchains
            &imageIndex,                                           // pImageIndices
            nullptr                                                // pResults
        };

        VkResult presentResult = vkQueuePresentKHR(vk.PresentationQueue, &presentInfo);
        if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR) {
          CreateSwapchain();
        } else if (presentResult != VK_SUCCESS) {
          throw std::runtime_error("Failed to present image!");
        }
      }

      currentFrame = (currentFrame + 1) % config.MaxImagesInFlight;
    }
  }
  LogTrace("Main application loop ended.");

  // =====
  // Cleanup
  // =====
  {
    vkDeviceWaitIdle(vk.Device);
    delete lightCube;
    delete cube;
    delete obj;
    for (U32 i = 0; i < 2; i++) {
      vkDestroyFence(vk.Device, vk.InFlightFences[i], nullptr);
      vkDestroySemaphore(vk.Device, vk.RenderFinishedSemaphores[i], nullptr);
      vkDestroySemaphore(vk.Device, vk.ImageAvailableSemaphores[i], nullptr);
    }
    vkFreeCommandBuffers(vk.Device, vk.GraphicsCommandPool,
                         static_cast<U32>(vk.CommandBuffers.size()), vk.CommandBuffers.data());
    vkDestroyRenderPass(vk.Device, vk.RenderPass, nullptr);
    vkDestroyImageView(vk.Device, vk.DepthImageView, nullptr);
    vkDestroyImage(vk.Device, vk.DepthImage, nullptr);
    vkFreeMemory(vk.Device, vk.DepthImageMemory, nullptr);
    for (U32 i = 0; i < vk.SwapchainImageCount; i++) {
      vkDestroyFramebuffer(vk.Device, vk.Framebuffers[i], nullptr);
      vkDestroyImageView(vk.Device, vk.SwapchainImageViews[i], nullptr);
    }
    vkDestroySwapchainKHR(vk.Device, vk.Swapchain, nullptr);
    vkDestroyCommandPool(vk.Device, vk.GraphicsCommandPool, nullptr);
    vkDestroyCommandPool(vk.Device, vk.TransferCommandPool, nullptr);
    vkDestroyDevice(vk.Device, nullptr);
    vkDestroySurfaceKHR(vk.Instance, vk.Surface, nullptr);
    {
      auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
          vk.Instance, "vkDestroyDebugUtilsMessengerEXT");
      if (func) {
        func(vk.Instance, vk.DebugMessenger, nullptr);
      }
    }
    vkDestroyInstance(vk.Instance, nullptr);
    DestroyWindow(app.Window);
  }
}

int main(int argc, char** argv) {
  try {
    Run();
  } catch (const std::exception& e) {
    LogFatal("!!! APPLICATION EXCEPTION: %s", e.what());
    __debugbreak();
    return 1;
  }

  return 0;
}