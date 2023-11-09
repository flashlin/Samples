Question: How to Add Versioning to ASP.NET Core Web API?
Answer:
Originally, the package was named Microsoft.AspNetCore.Mvc.Versioning. 
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

