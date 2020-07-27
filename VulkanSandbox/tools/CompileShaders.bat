@ECHO OFF
"C:\VulkanSDK\1.2.135.0\Bin\glslc.exe" ..\assets\shaders\Basic.vert -o ..\assets\shaders\Basic.vert.spv
"C:\VulkanSDK\1.2.135.0\Bin\glslc.exe" ..\assets\shaders\Basic.frag -o ..\assets\shaders\Basic.frag.spv
"C:\VulkanSDK\1.2.135.0\Bin\glslc.exe" ..\assets\shaders\Unlit.vert -o ..\assets\shaders\Unlit.vert.spv
"C:\VulkanSDK\1.2.135.0\Bin\glslc.exe" ..\assets\shaders\Unlit.frag -o ..\assets\shaders\Unlit.frag.spv
"C:\VulkanSDK\1.2.135.0\Bin\glslc.exe" ..\assets\shaders\Lit.vert -o ..\assets\shaders\Lit.vert.spv
"C:\VulkanSDK\1.2.135.0\Bin\glslc.exe" ..\assets\shaders\Lit.frag -o ..\assets\shaders\Lit.frag.spv
pause