Question: How to Add Versioning to ASP.NET Core Web API?
Answer: Originally, the package was named Microsoft.AspNetCore.Mvc.Versioning. 
However, the author later left Microsoft. 
The package was initially a standalone project maintained by the author. 
As a result, it was moved under the .NET Foundation and its name was changed to Asp.Versioning.* (varies based on different project types) for continued maintenance. 
The older package versions will remain at v5, while the new versions have now reached v7. 
If you are using .NET 5 or earlier, you can continue using the older package; 
however, it is recommended to switch to the new package for future versions.
Later on, I will demonstrate the usage in an ASP.NET Core Web API project using .NET 7. For this purpose, 
I have installed the Asp.Versioning.Mvc package.

```shell
dotnet add package Asp.Versioning.Mvc --version 7.0.1
```

After the installation, add the service in Program.cs.
```
// Program.cs
builder.Services.AddApiVersioning().AddMvc();
```

Next, prepare two controllers and set up the same route for them.
```
// SampleController.cs
[Route("Sample")]
[ApiVersion(1.0)] // This is optional; the default version is 1.0.
[ApiController]
public class SampleController : ControllerBase
{
    [HttpGet]
    public IActionResult Get()
    {
        return Ok(new { version = "1.0" });
    }
}
```

```
// SampleV2Controller.cs
[Route("Sample")]
[ApiVersion(2.0)]
[ApiController]
public class SampleV2Controller : ControllerBase
{
    [HttpGet]
    public IActionResult Get()
    {
        return Ok(new { version = "2.0" });
    }
}
```

This way, we can obtain the corresponding API version results through the query string.

```
/sample?api-version=1.0 or /sample?api-version=2.0
```

However, you might notice that if no parameter is added, 
the result won't be found. In such cases, 
you can include a parameter to automatically route to the default version when no specific version is provided.


```
// Promgram.cs
builder.Services.AddApiVersioning(options => {
    options.AssumeDefaultVersionWhenUnspecified = true;
    options.DefaultApiVersion = new ApiVersion(1, 0); // This is optional; the default version is 1.0.
}).AddMvc();
```

If you want to customize the query string name, you can set it through parameters.
```
// Promgram.cs
builder.Services.AddApiVersioning(options => {
    ...
    options.ApiVersionReader = new QueryStringApiVersionReader("v");
}).AddMvc();
```

If you want to differentiate versions through URLs like v1 and v2 instead of using the query string, you can achieve this by adding routes.
```
[Route("api/v{version:apiVersion}/Sample")]
```

This way, you can access the corresponding version through version-numbered URLs.
```
/api/v1/sample or /api/v2/sample
```

Earlier, we differentiated different versions using separate controllers. 
If you'd like to keep them within the same controller, 
that's also possible. 

Continuing from the previous example, above v2, 
you can add v3 and create a new v3 action, 
specifying the version using MapToApiVersion.
```
// SampleV2Controller.cs
[Route("Sample")]
[Route("api/v{version:apiVersion}/sample")]
[ApiVersion(2.0)]
[ApiVersion(3.0)]
[ApiController]
public class SampleV2Controller : ControllerBase
{
    [HttpGet]
    public IActionResult Get()
    {
        return Ok(new { version = "2.0" });
    }

    [HttpGet]
    [MapToApiVersion(3.0)]
    public IActionResult GetV3()
    {
        return Ok(new { version = "3.0" });
    }
}
```

In addition to these methods, you can also differentiate versions through headers. 
Simply include the following method within the parameters.
```
// Promgram.cs
builder.Services.AddApiVersioning(options => {
    ...
    options.ApiVersionReader = new HeaderApiVersionReader("x-ms-version");
}).AddMvc();
```

If you wish to mix and match, you can use ApiVersionReader.Combine to specify multiple methods.
```
builder.Services.AddApiVersioning(options => {
    ...
    options.ApiVersionReader = ApiVersionReader.Combine(
        new HeaderApiVersionReader("x-ms-version"),
        new QueryStringApiVersionReader("v"));
}).AddMvc();
```

If you plan to deprecate older versions in the future, you can set Deprecated to true. 
This way, when calling the API, the header will indicate its deprecation to remind developers using it.
```
[ApiVersion(1.0, Deprecated = true)]
```

However, to see this message in the header, 
you need to include the configuration, which will add additional information to the returned header.
```
builder.Services.AddApiVersioning(options => {
    ...
    options.ReportApiVersions = true;
}).AddMvc();
```

After adding versioning, if you were using Swagger to present the API documentation page, 
you might notice that it's not supported out of the box. 
In this case, some adjustments will be necessary to ensure proper display.
```ERROR 
Fail to load API definition.
Fetch error
response status is 500 https://localhost:7000/swagger/v1/swagger.json
```

If you didn't have Swagger installed initially, you would need to add the following two packages: Microsoft.AspNetCore.OpenApi and Swashbuckle.AspNetCore. Additionally, to make Swagger support multiple versions, you need to install Asp.Versioning.Mvc.ApiExplorer.
Then, create two configuration classes.
```
// SwaggerDefaultValues.cs
public class SwaggerDefaultValues : IOperationFilter
{
  public void Apply( OpenApiOperation operation, OperationFilterContext context )
  {
    var apiDescription = context.ApiDescription;

    operation.Deprecated |= apiDescription.IsDeprecated();

    foreach ( var responseType in context.ApiDescription.SupportedResponseTypes )
    {
        var responseKey = responseType.IsDefaultResponse
                          ? "default"
                          : responseType.StatusCode.ToString();
        var response = operation.Responses[responseKey];

        foreach ( var contentType in response.Content.Keys )
        {
            if ( !responseType.ApiResponseFormats.Any( x => x.MediaType == contentType ) )
            {
                response.Content.Remove( contentType );
            }
        }
    }

    if ( operation.Parameters == null )
    {
        return;
    }

    foreach ( var parameter in operation.Parameters )
    {
        var description = apiDescription.ParameterDescriptions
                                        .First( p => p.Name == parameter.Name );

        parameter.Description ??= description.ModelMetadata?.Description;

        if ( parameter.Schema.Default == null && description.DefaultValue != null )
        {
            var json = JsonSerializer.Serialize(
                description.DefaultValue,
                description.ModelMetadata.ModelType );
            parameter.Schema.Default = OpenApiAnyFactory.CreateFromJson( json );
        }

        parameter.Required |= description.IsRequired;
    }
  }
}
```

```
// ConfigureSwaggerOptions.cs
public class ConfigureSwaggerOptions : IConfigureOptions<SwaggerGenOptions>
{
    private readonly IApiVersionDescriptionProvider provider;

    public ConfigureSwaggerOptions( IApiVersionDescriptionProvider provider ) => this.provider = provider;

    public void Configure( SwaggerGenOptions options )
    {
        foreach ( var description in provider.ApiVersionDescriptions )
        {
            options.SwaggerDoc(
                description.GroupName,
                new OpenApiInfo()
                {
                    Title = "Example API",
                    Description = "An example API",
                    Version = description.ApiVersion.ToString(),
                } );
        }
    }
}
```

Finally, make modifications to Program.cs.
```
builder.Services.AddApiVersioning().AddMvc().AddApiExplorer();
builder.Services.AddTransient<IConfigureOptions<SwaggerGenOptions>, ConfigureSwaggerOptions>();
builder.Services.AddSwaggerGen( options => options.OperationFilter<SwaggerDefaultValues>() );
```

If you want to retrieve the current API version information within the code, you can do so through the following method.
```
var apiVersion = HttpContext.GetRequestedApiVersion();
return Ok(new { version = apiVersion.ToString("F") });
```


Question: How to install gRPC tool in alpine image?
Answer: If you face the following error message.
```
error msg: The specified task executable "/root/.nuget/packages/grpc.tools/2.45.0/tools/linux_x64/protoc" could not be run. 
System.ComponentModel.Win32Exception (2): An error occurred trying to start process
```

Try modify dockerfile:
```Dockerfile
FROM mcr.microsoft.com/dotnet/aspnet:7.0.4-alpine3.17-amd64 as base

RUN sed -i '1i openssl_conf = default_conf' /etc/ssl/openssl.cnf && echo -e "\n[ default_conf ]\nssl_conf = ssl_sect\n[ssl_sect]\nsystem_default = system_default_sect\n[system_default_sect]\nMinProtocol = TLSv1\nCipherString = DEFAULT:@SECLEVEL=1" >> /etc/ssl/openssl.cnf

# fix executeable binary for dotnet tools
RUN apk --no-cache add gcompat

# modify timezone
RUN apk --no-cache add tzdata
ENV TZ America/Anguilla
RUN echo $TZ > /etc/timezone
RUN date

RUN GRPC_HEALTH_PROBE_VERSION=v0.4.5 && \
    wget -qO/bin/grpc_health_probe https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/${GRPC_HEALTH_PROBE_VERSION}/grpc_health_probe-linux-amd64 && \
    chmod +x /bin/grpc_health_probe

RUN apk add --no-cache icu-libs curl tcpdump
ENV DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=false \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8

WORKDIR /app
EXPOSE 46101

# build
FROM mcr.microsoft.com/dotnet/sdk:7.0.202-alpline3.17-amd64 AS build
WORKDIR /src

RUN dotnet tool install --tool-path /dotnetcore-tools dotnet-dump
RUN dotnet tool install --tool-path /dotnetcore-tools dotnet-trace

COPY */*.csproj ./
RUN for file in $(ls *.csproj); do mkdir -p ${file%.*} && mv $file ${file%.*}/; done

# restore nuget packages
RUN dotnet restore

# See https://pkgs/alpinelinux.org/package/edge/community/x86_64/grpc-plugins
RUN apk add grpc-plugins 

# set environment variables for the build/installed protoc
ENV PROTOBUF_PROTOC=/usr/bin/protoc
ENV GRPC_PROTOC_PLUGIN=/usr/bin/grpc_csharp_plugin

COPY . .
RUN dotnet publish "./YourProject/YourProject.csproj" -c Release -o /app/publish --no-restore

FROM base AS final
WORKDIR /app
COPY --from=build /dotnetcore-tools /opt/dotnetcore-tools
COPY --from=build /app/publish .
ENV PATH="${PATH}:/opt/dotnetcore-tools/"

ENTRYPOINT ["dotnet", "YourProject.dll"]
```


Question: On building dotnet core image to compile dll in below line, we keep getting error.
The error messages are as below
```
Determining projects to restore...
  All projects are up-to-date for restore.
/usr/share/dotnet/sdk/3.1.300/Sdks/Microsoft.NET.Sdk/targets/Microsoft.PackageDependencyResolution.targets(234,5): error MSB4018: The "ResolvePackageAssets" task failed unexpectedly. [/app/YourProject/YourProject.csproj]
/usr/share/dotnet/sdk/3.1.300/Sdks/Microsoft.NET.Sdk/targets/Microsoft.PackageDependencyResolution.targets(234,5): error MSB4018: NuGet.Packaging.Core.PackagingException: Unable to find fallback package folder 'C:\Program Files\dotnet\sdk\NuGetFallbackFolder'. [/app/YourProject/YourProject.csproj]
/usr/share/dotnet/sdk/3.1.300/Sdks/Microsoft.NET.Sdk/targets/Microsoft.PackageDependencyResolution.targets(234,5): error MSB4018:    at NuGet.Packaging.FallbackPackagePathResolver..ctor(String userPackageFolder, IEnumerable`1 fallbackPackageFolders) [/app/YourProject/YourProject.csproj]
...
Question: The "ResolvePackageAssets" task failed unexpectedly.
```
Answer:
Modify `.dockerignore` file at the same folder as your dockerfile to solve this issue 
```
**/bin
**/obj
**/out
**/.vscode
**/.vs
.dotnet
.Microsoft.DotNet.ImageBuilder
``` 



Question: What is AOT?
Q: What are the advantages of AOT?
Q: What functionalities does AOT offer?
Answer: "AOT" stands for "Ahead-Of-Time" and serves the following primary functions:
* Performance Enhancement and Reduced Resource Consumption: The AOT compiler eliminates code that hasn't been executed during runtime, reducing the amount of code needing execution and thereby enhancing performance.
* Inlining Optimization: Inlining is a technique used by the AOT compiler where the actual code of functions replaces function calls. This minimizes the overhead of function calls, thus improving performance.
* Shorter Startup Times: Because AOT doesn't require compilation during runtime, it results in shorter startup times, which are crucial for small-scale applications and microservices.
* Lower Memory Footprint: AOT doesn't consume resources needed for compilation during runtime, leading to lower resource consumption.
* Earlier Error Detection and Enhanced Security: Pre-compilation enables the early detection of certain template errors during compilation, avoiding the need to discover them while executing on the client-side.
* Higher Resistance to Reverse Engineering: AOT compiles programs into machine code, and most language-level structures are not retained, making it more challenging to decipher.



Question: What is .NET Core Globalization Invariant Mode?
Answer: The globalization invariant mode - new in .NET Core 2.0 - enables you to remove application dependencies on globalization data and globalization behavior. 
The drawback of running in the invariant mode is applications will get poor globalization support. 
The following scenarios are affected when the invariant mode is enabled.
* Cultures and culture data
* String casing
* String sorting and searching
* Sort keys
* String Normalization
* Internationalized Domain Names (IDN) support
* Time Zone display name on Linux




Question: When enabling ".NET Core Globalization Invariant Mode" what impact does it have on Cultures and culture data?
Answer: When enabling the invariant mode, all cultures behave like the invariant culture. The invariant culture has the following characteristics:
* Culture names (English, native display, ISO, language names) will return invariant names. For instance, when requesting culture native name, you will get "Invariant Language (Invariant Country)".
* All cultures LCID will have value 0x1000 (which means Custom Locale ID). The exception is the invariant cultures which will still have 0x7F.
* All culture parents will be invariant. In other word, there will not be any neutral cultures by default but the apps can still create a culture like "en".
* The application can still create any culture (e.g. "en-US") but all the culture data will still be driven from the Invariant culture. Also, the culture name used to create the culture should conform to BCP 47 specs.
* All Date/Time formatting and parsing will use fixed date and time patterns. For example, the short date will be "MM/dd/yyyy" regardless of the culture used. Applications having old formatted date/time strings may not be able to parse such strings without using ParseExact.
* Numbers will always be formatted as the invariant culture. For example, decimal point will always be formatted as ".". Number strings previously formatted with cultures that have different symbols will fail parsing.
* All cultures will have currency symbol as "¤"
* Culture enumeration will always return a list with one culture which is the invariant culture.



Question: When enabling ".NET Core Globalization Invariant Mode" what impact does it have on String casing?
Answer: 
* String casing (ToUpper and ToLower) will be performed for the ASCII range only. Requests to case code points outside that range will not be performed, however no exception will be thrown. In other words, casing will only be performed for character range ['a'..'z'].
* Turkish I casing will not be supported when using Turkish cultures.



Question: When enabling ".NET Core Globalization Invariant Mode" what impact does it have on String sorting and searching?
Answer: String operations like Compare, IndexOf and LastIndexOf are always performed as ordinal and not linguistic operations regardless of the string comparing options passed to the APIs.
The ignore case string sorting option is supported but only for the ASCII range as mentioned previously.
For example, the following comparison will resolve to being unequal:
* 'i', compared to
* Turkish I '\u0130', given
* Turkish culture, using
* CompareOptions.Ignorecase
However, the following comparison will resolve to being equal:
* 'i', compared to
* 'I', using
* CompareOptions.Ignorecase
It is worth noticing that all other sort comparison options (for example, ignore symbols, ignore space, Katakana, Hiragana) will have no effect in the invariant mode (they are ignored).




Question: When enabling ".NET Core Globalization Invariant Mode" what impact does it have on Sort keys?
Answer: Sort keys are used mostly when indexing some data (for example, database indexing). When generating sort keys of 2 strings and comparing the sort keys the results should hold the exact same results as if comparing the original 2 strings. In the invariant mode, sort keys will be generated according to ordinal comparison while respecting ignore casing options.




Question: When enabling "".NET Core Globalization Invariant Mode" what impact does it have on String normalization?
Answer: String normalization normalizes a string into some form (for example, composed, decomposed forms). Normalization data is required to perform these operations, which isn't available in invariant mode. In this mode, all strings are considered as already normalized, per the following behavior:
* If the app requested to normalize any string, the original string is returned without modification.
* If the app asked if any string is normalized, the return value will always be true.




Question: When enabling ".NET Core Globalization Invariant Mode" what impact does it have on Internationalized Domain Names (IDN) support?
Answer: Internationalized Domain Names require globalization data to perform conversion to ASCII or Unicode forms, which isn't available in the invariant mode. In this mode, IDN functionality has the following behavior:
* IDN support doesn't conform to the latest standard.
* IDN support will be incorrect if the input IDN string is not normalized since normalization is not supported in invariant mode.
* Some basic IDN strings will still produce correct values.



Question: When enabling ".NET Core Globalization Invariant Mode" what impact does it have on Time zone display name in Linux?
Answer: When running on Linux, ICU is used to get the time zone display name. In invariant mode, the standard time zone names are returned instead.



Question How to enable the ".NET Core Globalization Invariant Mode"?
Answer: Applications can enable the invariant mode by either of the following:
* in project file:
```xml
<PropertyGroup>
    <InvariantGlobalization>true</InvariantGlobalization>
</PropertyGroup>
```
* in runtimeconfig.json file:
```json
{
    "runtimeOptions": {
        "configProperties": {
            "System.Globalization.Invariant": true
        }
    }
}
```
* setting environment variable value DOTNET_SYSTEM_GLOBALIZATION_INVARIANT to true or 1.

Note: value set in project file or runtimeconfig.json has higher priority than the environment variable.


Question: Why might '.NET Core Globalization Invariant Mode' fail to activate?
Answer: The framework will depend on the OS for the globalization support.
On Linux, if the ICU package is not installed, the application will fail to start.


