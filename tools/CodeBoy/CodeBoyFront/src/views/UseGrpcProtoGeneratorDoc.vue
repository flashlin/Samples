<template>
  <div class="container mx-auto px-4 py-8">
    <div class="max-w-4xl mx-auto">
      <!-- Header -->
      <div class="mb-8">
        <h1 class="text-3xl font-bold text-gray-100 mb-2">
          T1.GrpcProtoGenerator Usage Documentation
        </h1>
        <p class="text-gray-400">
          Learn how to use T1.GrpcProtoGenerator to automatically generate C# wrapper classes for your gRPC services.
        </p>
      </div>

      <!-- Documentation Content -->
      <div class="bg-gray-800 rounded-lg shadow-lg p-6 border border-gray-700">
        <div class="prose prose-lg max-w-none">
          <!-- Installation Section -->
          <section class="mb-8">
            <h2 class="text-2xl font-semibold text-gray-100 mb-4">Installation</h2>
            
            <p class="mb-4 text-gray-300">Install the NuGet package in your project:</p>
            
            <div class="bg-gray-900 rounded-lg p-4 mb-4 border border-gray-600">
              <code class="text-sm text-green-400">dotnet add package T1.GrpcProtoGenerator</code>
            </div>
            
            <p class="mb-4 text-gray-300">Or via Package Manager Console:</p>
            
            <div class="bg-gray-900 rounded-lg p-4 mb-6 border border-gray-600">
              <code class="text-sm text-green-400">Install-Package T1.GrpcProtoGenerator</code>
            </div>
          </section>

          <!-- Usage Section -->
          <section class="mb-8">
            <h2 class="text-2xl font-semibold text-gray-100 mb-4">Usage</h2>
            
            <p class="mb-4 text-gray-300">Given a .proto file:</p>
            
            <pre class="bg-gray-900 text-green-400 rounded-lg p-4 mb-6 overflow-x-auto"><code>syntax = "proto3";

service Greeter {
  rpc SayHello (HelloRequest) returns (HelloReply);
}

message HelloRequest {
  string name = 1;
}

message HelloReply {
  string message = 1;
}</code></pre>

            <p class="mb-4 text-gray-300">Add .proto files to your project and ensure they are included as &lt;Protobuf&gt; items in your .csproj:</p>
            
            <pre class="bg-gray-900 text-blue-400 rounded-lg p-4 mb-6 overflow-x-auto"><code>&lt;ItemGroup&gt;
    &lt;Protobuf Include="Protos\greet.proto" GrpcServices="Both" ProtoRoot="Protos" /&gt;
    &lt;Protobuf Include="Protos\Messages\requests.proto" GrpcServices="None" ProtoRoot="Protos" /&gt;
    &lt;Protobuf Include="Protos\Messages\responses.proto" GrpcServices="None" ProtoRoot="Protos" /&gt;
    &lt;AdditionalFiles Include="Protos\**\*.proto" /&gt;
&lt;/ItemGroup&gt;

&lt;ItemGroup&gt;
    &lt;PackageReference Include="Grpc.AspNetCore" Version="2.64.0"/&gt;
    &lt;PackageReference Include="Google.Api.CommonProtos" Version="2.15.0"/&gt;
    &lt;PackageReference Include="Grpc.AspNetCore.Web" Version="2.64.0"/&gt;
&lt;/ItemGroup&gt;

&lt;ItemGroup&gt;
    &lt;Compile Remove="Generated\**" /&gt;
&lt;/ItemGroup&gt;</code></pre>

            <p class="mb-4 text-gray-300">Build your project - the source generator will automatically create wrapper classes for your gRPC services.</p>
            
            <pre class="bg-gray-900 text-yellow-400 rounded-lg p-4 mb-6 overflow-x-auto"><code>public class GreeterService : IGreeterGrpcService
{
    public Task&lt;HelloReplyGrpcDto&gt; SayHello(HelloRequestGrpcDto request)
    {
        var response = new HelloReplyGrpcDto
        {
            Message = $"Hello {request.Name}!"
        };
        return Task.FromResult(response);
    }
}</code></pre>

            <p class="mb-4 text-gray-300">Use the generated server wrappers:</p>
            
            <pre class="bg-gray-900 text-cyan-400 rounded-lg p-4 mb-6 overflow-x-auto"><code>builder.Services.AddGrpc();
builder.Services.AddScoped&lt;IGreeterGrpcService, GreeterService&gt;();

var app = builder.Build();
// Configure the HTTP request pipeline.
app.MapGrpcService&lt;GreeterNativeGrpcService&gt;();</code></pre>

            <p class="mb-4 text-gray-300">Use the generated client wrappers:</p>
            
            <pre class="bg-gray-900 text-purple-400 rounded-lg p-4 mb-6 overflow-x-auto"><code>var services = new ServiceCollection();
// Configure gRPC server settings
services.Configure&lt;GreeterGrpcConfig&gt;(config =>
{
    config.ServerUrl = "https://localhost:7001"; // Your gRPC server address
});
// Register gRPC SDK using auto-generated extension method
services.AddGreeterGrpcSdk();
// Generated wrapper provides a clean, easy-to-use API
var client = sp.GetRequiredService&lt;IGreeterGrpcClient&gt;();
var request = new HelloRequestGrpcDto 
{ 
    Name = "World from Consumer App" 
};
var response = await grpcClient.SayHelloAsync(request);
Console.WriteLine(response.Message);</code></pre>
          </section>

          <!-- Key Features Section -->
          <section class="mb-8">
            <h2 class="text-2xl font-semibold text-gray-100 mb-4">Key Features</h2>
            
            <ul class="list-disc list-inside space-y-2 text-gray-300">
              <li>Automatic generation of C# wrapper classes from .proto files</li>
              <li>Support for both client and server-side gRPC services</li>
              <li>Integration with ASP.NET Core dependency injection</li>
              <li>Clean separation between generated and user code</li>
              <li>Support for complex message types and nested structures</li>
            </ul>
          </section>

          <!-- Notes Section -->
          <section class="mb-8">
            <h2 class="text-2xl font-semibold text-gray-100 mb-4">Important Notes</h2>
            
            <div class="bg-blue-900/20 border-l-4 border-blue-400 p-4 mb-4">
              <div class="flex">
                <div class="flex-shrink-0">
                  <svg class="h-5 w-5 text-blue-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd" />
                  </svg>
                </div>
                <div class="ml-3">
                  <p class="text-sm text-blue-300">
                    Make sure to exclude the Generated folder from compilation to avoid conflicts.
                  </p>
                </div>
              </div>
            </div>

            <div class="bg-yellow-900/20 border-l-4 border-yellow-400 p-4 mb-4">
              <div class="flex">
                <div class="flex-shrink-0">
                  <svg class="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
                  </svg>
                </div>
                <div class="ml-3">
                  <p class="text-sm text-yellow-300">
                    The source generator runs during build time. Clean and rebuild if changes to .proto files are not reflected.
                  </p>
                </div>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
// Component for displaying T1.GrpcProtoGenerator usage documentation
</script>

<style scoped>
/* Additional styling for code blocks */
pre {
  font-family: 'Courier New', Courier, monospace;
  white-space: pre-wrap;
  word-wrap: break-word;
}

code {
  font-family: 'Courier New', Courier, monospace;
}

/* Ensure proper scrolling for code blocks */
pre code {
  display: block;
  padding: 0;
  margin: 0;
  overflow: visible;
}

/* Custom scrollbar for code blocks */
pre::-webkit-scrollbar {
  height: 8px;
}

pre::-webkit-scrollbar-track {
  background: #1f2937;
}

pre::-webkit-scrollbar-thumb {
  background: #4b5563;
  border-radius: 4px;
}

pre::-webkit-scrollbar-thumb:hover {
  background: #6b7280;
}
</style>
