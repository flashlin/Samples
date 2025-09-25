using CodeBoyLib.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using T1.Standard.IO;

namespace CodeBoyLib.Services
{
    public class SwaggerClientCodeGenerator
    {
        public string Generate(string sdkName, SwaggerApiInfo apiInfo)
        {
            var output = new IndentStringBuilder();
            
            // Add using statements
            output.WriteLine("using System;");
            output.WriteLine("using System.Collections.Generic;");
            output.WriteLine("using System.Linq;");
            output.WriteLine("using System.Net.Http;");
            output.WriteLine("using System.Text;");
            output.WriteLine("using System.Threading.Tasks;");
            output.WriteLine("using Microsoft.Extensions.Http;");
            output.WriteLine("using Microsoft.Extensions.Options;");
            output.WriteLine("using System.Text.Json;");
            output.WriteLine("using System.Text.Json.Serialization;");
            output.WriteLine("using System.ComponentModel.DataAnnotations;");
            output.WriteLine();

            // Add namespace
            output.WriteLine($"namespace {sdkName}SDK");
            output.WriteLine("{");
            output.Indent++;

            // Generate model classes from definitions
            foreach (var classDef in apiInfo.ClassDefinitions.Values)
            {
                // Skip numeric-only enums - they will be treated as int in DTOs
                if (!classDef.IsNumericEnum)
                {
                    GenerateModelClass(output, classDef);
                }
            }

            // Generate the configuration class
            GenerateConfigClass(output, sdkName);

            // Generate the main client class
            GenerateClientClass(output, sdkName, apiInfo.Endpoints);

            output.Indent--;
            output.WriteLine("}");

            return output.ToString();
        }

        private bool IsPrimitiveType(string type)
        {
            var primitiveTypes = new HashSet<string>
            {
                "string", "int", "long", "float", "double", "decimal", "bool", "DateTime", "Guid", "object", "void"
            };
            return primitiveTypes.Contains(type) || type.StartsWith("List<");
        }

        private void GenerateConfigClass(IndentStringBuilder output, string sdkName)
        {
            var configClassName = $"{sdkName}ClientConfig";
            
            output.WriteLine($"/// <summary>");
            output.WriteLine($"/// Configuration settings for {sdkName} API client");
            output.WriteLine($"/// </summary>");
            output.WriteLine($"public class {configClassName}");
            output.WriteLine("{");
            output.Indent++;
            
            output.WriteLine("/// <summary>");
            output.WriteLine("/// Base URL for the API");
            output.WriteLine("/// </summary>");
            output.WriteLine("public string BaseUrl { get; set; } = string.Empty;");
            output.WriteLine();
            
            output.WriteLine("/// <summary>");
            output.WriteLine("/// HTTP client name for dependency injection (optional)");
            output.WriteLine("/// </summary>");
            output.WriteLine("public string? HttpClientName { get; set; }");
            output.WriteLine();
            
            output.WriteLine("/// <summary>");
            output.WriteLine("/// Request timeout in seconds (optional)");
            output.WriteLine("/// </summary>");
            output.WriteLine("public int? TimeoutSeconds { get; set; }");
            
            output.Indent--;
            output.WriteLine("}");
            output.WriteLine();
        }

        private void GenerateModelClass(IndentStringBuilder output, ClassDefinition classDef)
        {
            output.WriteLine($"/// <summary>");
            output.WriteLine($"/// {classDef.Description}");
            output.WriteLine($"/// </summary>");
            
            if (classDef.IsEnum)
            {
                GenerateEnumClass(output, classDef);
            }
            else
            {
                GenerateObjectClass(output, classDef);
            }
        }

        private void GenerateEnumClass(IndentStringBuilder output, ClassDefinition classDef)
        {
            output.WriteLine($"public enum {classDef.Name}");
            output.WriteLine("{");
            output.Indent++;
            
            for (int i = 0; i < classDef.EnumValues.Count; i++)
            {
                var enumValue = classDef.EnumValues[i];
                var sanitizedValue = SanitizeEnumValue(enumValue);
                
                if (i == classDef.EnumValues.Count - 1)
                {
                    output.WriteLine($"{sanitizedValue}");
                }
                else
                {
                    output.WriteLine($"{sanitizedValue},");
                }
            }
            
            output.Indent--;
            output.WriteLine("}");
            output.WriteLine();
        }

        private void GenerateObjectClass(IndentStringBuilder output, ClassDefinition classDef)
        {
            output.WriteLine($"public class {classDef.Name}");
            output.WriteLine("{");
            output.Indent++;
            
            foreach (var property in classDef.Properties)
            {
                GenerateProperty(output, property);
            }
            
            output.Indent--;
            output.WriteLine("}");
            output.WriteLine();
        }

        private void GenerateProperty(IndentStringBuilder output, ClassProperty property)
        {
            // Add XML documentation
            if (!string.IsNullOrEmpty(property.Description))
            {
                output.WriteLine($"/// <summary>");
                output.WriteLine($"/// {property.Description}");
                output.WriteLine($"/// </summary>");
            }

            // Add validation attributes
            if (property.IsRequired)
            {
                output.WriteLine($"[Required]");
            }

            // Add JSON property attribute
            output.WriteLine($"[JsonPropertyName(\"{property.Name}\")]");

            // Generate property declaration
            var defaultValue = GetDefaultValueString(property);
            if (!string.IsNullOrEmpty(defaultValue))
            {
                output.WriteLine($"public {property.Type} {PascalCase(property.Name)} {{ get; set; }} = {defaultValue};");
            }
            else
            {
                if (property.Type.EndsWith("?") || !property.IsRequired)
                {
                    output.WriteLine($"public {property.Type} {PascalCase(property.Name)} {{ get; set; }}");
                }
                else
                {
                    output.WriteLine($"public {property.Type} {PascalCase(property.Name)} {{ get; set; }} = default!;");
                }
            }
            
            output.WriteLine();
        }

        private string GetDefaultValueString(ClassProperty property)
        {
            if (property.DefaultValue == null) 
                return string.Empty;

            return property.Type switch
            {
                "string" => $"\"{property.DefaultValue}\"",
                "int" or "long" or "float" or "double" or "decimal" => property.DefaultValue.ToString() ?? string.Empty,
                "bool" => property.DefaultValue.ToString()?.ToLower() ?? string.Empty,
                _ when property.Type.StartsWith("List<") => $"new {property.Type}()",
                _ => property.DefaultValue.ToString() ?? string.Empty
            };
        }

        private string SanitizeEnumValue(string enumValue)
        {
            // Remove special characters and ensure it starts with a letter
            var sanitized = new StringBuilder();
            bool firstChar = true;

            foreach (char c in enumValue)
            {
                if (char.IsLetterOrDigit(c))
                {
                    if (firstChar && char.IsDigit(c))
                    {
                        sanitized.Append('_');
                    }
                    sanitized.Append(char.ToUpper(c));
                    firstChar = false;
                }
                else if (!firstChar)
                {
                    if (sanitized.Length > 0 && sanitized[sanitized.Length - 1] != '_')
                    {
                        sanitized.Append('_');
                    }
                }
            }

            var result = sanitized.ToString().TrimEnd('_');
            return string.IsNullOrEmpty(result) ? "UNKNOWN" : result;
        }

        private string PascalCase(string input)
        {
            if (string.IsNullOrEmpty(input))
                return input;

            return char.ToUpper(input[0]) + input[1..];
        }

        private void GenerateClientClass(IndentStringBuilder output, string sdkName, List<SwaggerEndpoint> endpoints)
        {
            var className = $"{sdkName}Client";
            
            // Generate class declaration and fields
            GenerateClientClassDeclaration(output, sdkName, className);
            
            // Generate constructors
            GeneratePrimaryConstructor(output, sdkName, className);
            GenerateBackwardCompatibilityConstructor(output, sdkName, className);

            // Generate methods for each endpoint
            foreach (var endpoint in endpoints)
            {
                GenerateEndpointMethod(output, endpoint);
            }

            // Helper methods
            GenerateHelperMethods(output);

            output.Indent--;
            output.WriteLine("}");
        }

        private void GenerateClientClassDeclaration(IndentStringBuilder output, string sdkName, string className)
        {
            output.WriteLine($"/// <summary>");
            output.WriteLine($"/// HTTP client for {sdkName} API");
            output.WriteLine($"/// </summary>");
            output.WriteLine($"public class {className}");
            output.WriteLine("{");
            output.Indent++;
            
            output.WriteLine("private readonly HttpClient _httpClient;");
            output.WriteLine($"private readonly {sdkName}ClientConfig _config;");
            output.WriteLine();
        }

        private void GeneratePrimaryConstructor(IndentStringBuilder output, string sdkName, string className)
        {
            output.WriteLine($"/// <summary>");
            output.WriteLine($"/// Initializes a new instance of {className}");
            output.WriteLine($"/// </summary>");
            output.WriteLine($"/// <param name=\"httpClientFactory\">HTTP client factory</param>");
            output.WriteLine($"/// <param name=\"config\">Configuration options</param>");
            output.WriteLine($"public {className}(IHttpClientFactory httpClientFactory, IOptions<{sdkName}ClientConfig> config)");
            output.WriteLine("{");
            output.Indent++;
            
            output.WriteLine("_config = config.Value ?? throw new ArgumentNullException(nameof(config));");
            output.WriteLine("if (string.IsNullOrEmpty(_config.BaseUrl))");
            output.Indent++;
            output.WriteLine("throw new ArgumentException(\"BaseUrl must be configured\", nameof(config));");
            output.Indent--;
            output.WriteLine();
            
            output.WriteLine("if (!string.IsNullOrEmpty(_config.HttpClientName))");
            output.WriteLine("{");
            output.Indent++;
            output.WriteLine("_httpClient = httpClientFactory.CreateClient(_config.HttpClientName);");
            output.Indent--;
            output.WriteLine("}");
            output.WriteLine("else");
            output.WriteLine("{");
            output.Indent++;
            output.WriteLine("_httpClient = httpClientFactory.CreateClient();");
            output.Indent--;
            output.WriteLine("}");
            output.WriteLine();
            
            output.WriteLine("if (_config.TimeoutSeconds.HasValue)");
            output.WriteLine("{");
            output.Indent++;
            output.WriteLine("_httpClient.Timeout = TimeSpan.FromSeconds(_config.TimeoutSeconds.Value);");
            output.Indent--;
            output.WriteLine("}");
            
            output.Indent--;
            output.WriteLine("}");
            output.WriteLine();
        }

        private void GenerateBackwardCompatibilityConstructor(IndentStringBuilder output, string sdkName, string className)
        {
            output.WriteLine($"/// <summary>");
            output.WriteLine($"/// Initializes a new instance of {className} (backward compatibility)");
            output.WriteLine($"/// </summary>");
            output.WriteLine($"/// <param name=\"httpClient\">HTTP client instance</param>");
            output.WriteLine($"/// <param name=\"baseUrl\">Base URL for the API</param>");
            output.WriteLine($"public {className}(HttpClient httpClient, string baseUrl)");
            output.WriteLine("{");
            output.Indent++;
            
            output.WriteLine("_httpClient = httpClient ?? throw new ArgumentNullException(nameof(httpClient));");
            output.WriteLine($"_config = new {sdkName}ClientConfig {{ BaseUrl = baseUrl.TrimEnd('/') }};");
            
            output.Indent--;
            output.WriteLine("}");
            output.WriteLine();
        }

        private void GenerateEndpointMethod(IndentStringBuilder output, SwaggerEndpoint endpoint)
        {
            var methodName = SanitizeMethodName(endpoint.Path);
            var returnType = GetReturnType(endpoint.ResponseType);
            
            output.WriteLine($"/// <summary>");
            output.WriteLine($"/// {endpoint.Summary}");
            if (!string.IsNullOrEmpty(endpoint.Description))
            {
                output.WriteLine($"/// {endpoint.Description}");
            }
            output.WriteLine($"/// </summary>");

            // Generate method parameters
            var methodParams = new List<string>();
            var pathParams = endpoint.Parameters.Where(p => p.Location == "path").ToList();
            var queryParams = endpoint.Parameters.Where(p => p.Location == "query").ToList();
            var bodyParams = endpoint.Parameters.Where(p => p.Location == "body").ToList();
            var headerParams = endpoint.Parameters.Where(p => p.Location == "header").ToList();

            // Add path parameters
            foreach (var param in pathParams)
            {
                methodParams.Add($"{param.Type} {param.Name}");
                output.WriteLine($"/// <param name=\"{param.Name}\">{param.Description}</param>");
            }

            // Add body parameter if exists
            if (bodyParams.Any())
            {
                var bodyParam = bodyParams.First();
                methodParams.Add($"{bodyParam.Type} {bodyParam.Name}");
                output.WriteLine($"/// <param name=\"{bodyParam.Name}\">{bodyParam.Description}</param>");
            }

            // Add query parameters as optional
            foreach (var param in queryParams)
            {
                var paramType = param.IsRequired ? param.Type : $"{param.Type}?";
                var defaultValue = param.IsRequired ? "" : " = null";
                methodParams.Add($"{paramType} {param.Name}{defaultValue}");
                output.WriteLine($"/// <param name=\"{param.Name}\">{param.Description}</param>");
            }

            // Add header parameters as optional
            foreach (var param in headerParams)
            {
                var paramType = param.IsRequired ? param.Type : $"{param.Type}?";
                var defaultValue = param.IsRequired ? "" : " = null";
                methodParams.Add($"{paramType} {param.Name}{defaultValue}");
                output.WriteLine($"/// <param name=\"{param.Name}\">{param.Description}</param>");
            }

            output.WriteLine($"/// <returns>{returnType}</returns>");
            output.WriteLine($"public async {returnType} {methodName}({string.Join(", ", methodParams)})");
            output.WriteLine("{");
            output.Indent++;

            // Build URL
            var urlBuilder = new StringBuilder();
            urlBuilder.Append($"var url = $\"{endpoint.Path}\"");
            
            // Replace path parameters
            foreach (var param in pathParams)
            {
                urlBuilder.Replace($"{{{param.Name}}}", $"{{{param.Name}}}");
            }
            urlBuilder.Append(";");
            output.WriteLine(urlBuilder.ToString());

            // Add query parameters
            if (queryParams.Any())
            {
                output.WriteLine("var queryParams = new List<string>();");
                foreach (var param in queryParams)
                {
                    if (param.IsRequired)
                    {
                        output.WriteLine($"queryParams.Add($\"{param.Name}={{Uri.EscapeDataString({param.Name}.ToString())}}\");");
                    }
                    else
                    {
                        output.WriteLine($"if ({param.Name} != null)");
                        output.Indent++;
                        output.WriteLine($"queryParams.Add($\"{param.Name}={{Uri.EscapeDataString({param.Name}.ToString())}}\");");
                        output.Indent--;
                    }
                }
                output.WriteLine("if (queryParams.Any())");
                output.Indent++;
                output.WriteLine("url += \"?\" + string.Join(\"&\", queryParams);");
                output.Indent--;
                output.WriteLine();
            }

            // Create HTTP request
            var httpMethod = endpoint.HttpMethod.ToUpper() switch
            {
                "GET" => "HttpMethod.Get",
                "POST" => "HttpMethod.Post",
                "PUT" => "HttpMethod.Put",
                "DELETE" => "HttpMethod.Delete",
                "PATCH" => "HttpMethod.Patch",
                "HEAD" => "HttpMethod.Head",
                "OPTIONS" => "HttpMethod.Options",
                _ => $"new HttpMethod(\"{endpoint.HttpMethod.ToUpper()}\")"
            };
            output.WriteLine($"var request = new HttpRequestMessage({httpMethod}, _config.BaseUrl + url);");

            // Add headers
            foreach (var param in headerParams)
            {
                if (param.IsRequired)
                {
                    output.WriteLine($"request.Headers.Add(\"{param.Name}\", {param.Name}.ToString());");
                }
                else
                {
                    output.WriteLine($"if ({param.Name} != null)");
                    output.Indent++;
                    output.WriteLine($"request.Headers.Add(\"{param.Name}\", {param.Name}.ToString());");
                    output.Indent--;
                }
            }

            // Add body content for POST/PUT/PATCH
            if (bodyParams.Any() && (endpoint.HttpMethod.ToUpper() == "POST" || endpoint.HttpMethod.ToUpper() == "PUT" || endpoint.HttpMethod.ToUpper() == "PATCH"))
            {
                var bodyParam = bodyParams.First();
                output.WriteLine($"if ({bodyParam.Name} != null)");
                output.WriteLine("{");
                output.Indent++;
                output.WriteLine($"var jsonContent = JsonSerializer.Serialize({bodyParam.Name});");
                output.WriteLine("request.Content = new StringContent(jsonContent, Encoding.UTF8, \"application/json\");");
                output.Indent--;
                output.WriteLine("}");
            }

            output.WriteLine();
            output.WriteLine("var response = await _httpClient.SendAsync(request);");
            output.WriteLine("response.EnsureSuccessStatusCode();");
            output.WriteLine();

            // Handle response
            if (returnType == "Task")
            {
                output.WriteLine("// No return value expected");
            }
            else if (returnType == "Task<string>")
            {
                output.WriteLine("return await response.Content.ReadAsStringAsync();");
            }
            else
            {
                output.WriteLine("var responseContent = await response.Content.ReadAsStringAsync();");
                output.WriteLine();
                output.WriteLine("if (string.IsNullOrEmpty(responseContent))");
                output.Indent++;
                
                if (returnType.StartsWith("Task<List<"))
                {
                    var innerType = returnType.Substring(5, returnType.Length - 6); // Remove "Task<" and ">"
                    output.WriteLine($"return null;");
                    output.Indent--;
                    output.WriteLine();
                    output.WriteLine($"return JsonSerializer.Deserialize<{innerType}>(responseContent);");
                }
                else if (returnType.StartsWith("Task<"))
                {
                    var innerType = returnType.Substring(5, returnType.Length - 6); // Remove "Task<" and ">"
                    
                    // Since all return types are now nullable, we can directly return the deserialized result
                    output.WriteLine($"return null;");
                    output.Indent--;
                    output.WriteLine();
                    output.WriteLine($"return JsonSerializer.Deserialize<{innerType}>(responseContent);");
                }
                else
                {
                    output.Indent--;
                }
            }

            output.Indent--;
            output.WriteLine("}");
            output.WriteLine();
        }

        private void GenerateHelperMethods(IndentStringBuilder output)
        {
            output.WriteLine("/// <summary>");
            output.WriteLine("/// Dispose resources");
            output.WriteLine("/// </summary>");
            output.WriteLine("public void Dispose()");
            output.WriteLine("{");
            output.Indent++;
            
            output.WriteLine("_httpClient?.Dispose();");
            
            output.Indent--;
            output.WriteLine("}");
            output.WriteLine();
        }

        private string GetReturnType(ResponseType responseType)
        {
            if (responseType == null || string.IsNullOrEmpty(responseType.Type) || responseType.Type == "void")
            {
                return "Task";
            }

            if (responseType.IsArray)
            {
                // Arrays (List<T>) can be null, so always add ?
                return $"Task<List<{responseType.Type}>?>";
            }

            // For reference types, add ? since they can return null when response is empty
            // For value types that already have ?, keep them as is  
            if (!responseType.Type.EndsWith("?") && !IsPrimitiveType(responseType.Type))
            {
                return $"Task<{responseType.Type}?>";
            }

            return $"Task<{responseType.Type}>";
        }

        private string SanitizeMethodName(string endpointPath)
        {
            if (string.IsNullOrEmpty(endpointPath))
            {
                return "UnknownEndpoint";
            }

            // Extract the last segment from the path
            // e.g., "/user/{id}/profile" -> "profile"
            // e.g., "/pet" -> "pet"
            // e.g., "/store/inventory" -> "inventory"
            
            var pathSegments = endpointPath.Split('/', StringSplitOptions.RemoveEmptyEntries);
            
            // Find the last segment that is not a parameter (not in {})
            string? lastSegment = null;
            for (int i = pathSegments.Length - 1; i >= 0; i--)
            {
                var segment = pathSegments[i];
                if (!segment.StartsWith("{") && !segment.EndsWith("}"))
                {
                    lastSegment = segment;
                    break;
                }
            }
            
            // If no valid segment found, use a fallback
            if (string.IsNullOrEmpty(lastSegment))
            {
                return "UnknownEndpoint";
            }
            
            // Clean the segment and capitalize first letter only
            var cleanSegment = CleanSegmentName(lastSegment);
            
            // Ensure it starts with a letter and capitalize first letter only
            if (string.IsNullOrEmpty(cleanSegment))
            {
                return "UnknownEndpoint";
            }
            
            return char.ToUpper(cleanSegment[0]) + cleanSegment.Substring(1);
        }
        
        private string CleanSegmentName(string segment)
        {
            if (string.IsNullOrEmpty(segment))
                return string.Empty;
                
            var result = new StringBuilder();
            
            foreach (char c in segment)
            {
                if (char.IsLetterOrDigit(c))
                {
                    result.Append(c);
                }
            }
            
            var cleanedResult = result.ToString();
            
            // If it starts with a digit, prepend with "Endpoint"
            if (!string.IsNullOrEmpty(cleanedResult) && char.IsDigit(cleanedResult[0]))
            {
                cleanedResult = "Endpoint" + cleanedResult;
            }
            
            return string.IsNullOrEmpty(cleanedResult) ? "UnknownEndpoint" : cleanedResult;
        }
    }
}
