## 运行说明
### 运行环境
1. C++编译器
2. CMake
### 运行步骤
1. 创建build目录（如果没有）
2. 执行 `cd build && cmake ..` 生成对应平台的make文件
3. Linux/Unix/MacOS 执行 `make`命令编译出二进制文件；Windows 执行 `msbuild ObjectDetection.sln` 命令编译出二进制文件
4. 运行二进制文件ObjectDetection(.exe)