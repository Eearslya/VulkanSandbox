@ECHO OFF
"C:\VulkanSDK\1.2.135.0\Bin\glslc.exe" ..\assets\shaders\Basic.vert -o ..\assets\shaders\Basic.vert.spv
"C:\VulkanSDK\1.2.135.0\Bin\glslc.exe" ..\assets\shaders\Basic.frag -o ..\assets\shaders\Basic.frag.spv
pause