安裝 dev c++ 5.8.2
https://sourceforge.net/projects/orwelldevcpp/files/Setup%20Releases/Dev-Cpp%205.8.2%20TDM-GCC%204.8.1%20Setup.exe/download

git clone https://github.com/emscripten-core/emsdk.git

進入 miniconda python 環境
./emsdk.bat install latest
./emsdk.bat activate latest
./emsdk_env.bat

./emsdk.bat install 1.38.45-64bit

啟動環境
./emcmdprompt.bat



dotnet new blazorwasm -o MyProject
dotnet new classlib -f net7.0 -o system-io
編輯 system-io.csproj 文件，並添加以下項目
```
<PropertyGroup>
    <BlazorWebAssemblyEnableLinking>true</BlazorWebAssemblyEnableLinking>
</PropertyGroup>
```