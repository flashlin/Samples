using HtmlAgilityPack;
using MakeSwaggerSDK.Models;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace MakeSwaggerSDK.Services
{
    public class SwaggerUiParser
    {
        private readonly HttpClient _httpClient;

        public SwaggerUiParser()
        {
            _httpClient = new HttpClient();
        }

        public async Task<List<SwaggerEndpoint>> Parse(string swaggerUrl)
        {
            try
            {
                // Check if the URL is already a JSON endpoint
                if (swaggerUrl.EndsWith(".json") || swaggerUrl.Contains("swagger.json") || swaggerUrl.Contains("api-docs"))
                {
                    Console.WriteLine($"Direct JSON URL detected: {swaggerUrl}");
                    var jsonContent = await _httpClient.GetStringAsync(swaggerUrl);
                    return ParseSwaggerJson(jsonContent);
                }

                // Get the HTML content from Swagger UI
                var htmlContent = await _httpClient.GetStringAsync(swaggerUrl);
                
                // Check if response is already JSON (not HTML)
                if (IsJsonContent(htmlContent))
                {
                    Console.WriteLine("Response is JSON content, parsing directly...");
                    return ParseSwaggerJson(htmlContent);
                }
                
                // Extract Swagger JSON URL from HTML
                var swaggerJsonUrl = ExtractSwaggerJsonUrl(htmlContent, swaggerUrl);
                if (string.IsNullOrEmpty(swaggerJsonUrl))
                {
                    throw new Exception("Could not find Swagger JSON URL in the HTML content");
                }

                Console.WriteLine($"Found Swagger JSON URL: {swaggerJsonUrl}");

                // Get the Swagger JSON specification
                var swaggerJsonContent = await _httpClient.GetStringAsync(swaggerJsonUrl);
                
                // Parse the Swagger JSON
                return ParseSwaggerJson(swaggerJsonContent);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error parsing Swagger UI: {ex.Message}");
                throw;
            }
        }

        private bool IsJsonContent(string content)
        {
            if (string.IsNullOrWhiteSpace(content))
                return false;

            var trimmed = content.Trim();
            return (trimmed.StartsWith("{") && trimmed.EndsWith("}")) ||
                   (trimmed.StartsWith("[") && trimmed.EndsWith("]"));
        }

        private string ExtractSwaggerJsonUrl(string htmlContent, string baseUrl)
        {
            // Common patterns for Swagger JSON URL in Swagger UI
            var patterns = new[]
            {
                @"url\s*:\s*[""']([^""']+\.json)[""']",
                @"spec\s*:\s*[""']([^""']+\.json)[""']",
                @"[""']([^""']*swagger\.json[^""']*)[""']",
                @"[""']([^""']*api-docs[^""']*)[""']",
                @"configUrl\s*:\s*[""']([^""']+)[""']"
            };

            foreach (var pattern in patterns)
            {
                var match = Regex.Match(htmlContent, pattern, RegexOptions.IgnoreCase);
                if (match.Success)
                {
                    var url = match.Groups[1].Value;
                    
                    // If it's a relative URL, make it absolute
                    if (!url.StartsWith("http"))
                    {
                        var baseUri = new Uri(baseUrl);
                        if (url.StartsWith("/"))
                        {
                            url = $"{baseUri.Scheme}://{baseUri.Host}:{baseUri.Port}{url}";
                        }
                        else
                        {
                            url = $"{baseUrl.TrimEnd('/')}/{url.TrimStart('/')}";
                        }
                    }
                    
                    return url;
                }
            }

            // Try to find any JSON URLs that might be Swagger specs
            var jsonUrlPattern = @"[""']([^""']*\.json[^""']*)[""']";
            var jsonMatches = Regex.Matches(htmlContent, jsonUrlPattern, RegexOptions.IgnoreCase);
            
            foreach (Match match in jsonMatches)
            {
                var url = match.Groups[1].Value;
                if (url.Contains("swagger") || url.Contains("api-docs") || url.Contains("openapi"))
                {
                    if (!url.StartsWith("http"))
                    {
                        var baseUri = new Uri(baseUrl);
                        if (url.StartsWith("/"))
                        {
                            url = $"{baseUri.Scheme}://{baseUri.Host}:{baseUri.Port}{url}";
                        }
                        else
                        {
                            url = $"{baseUrl.TrimEnd('/')}/{url.TrimStart('/')}";
                        }
                    }
                    return url;
                }
            }

            return string.Empty;
        }

        private List<SwaggerEndpoint> ParseSwaggerJson(string jsonContent)
        {
            var endpoints = new List<SwaggerEndpoint>();
            
            try
            {
                var swaggerDoc = JObject.Parse(jsonContent);
                var paths = swaggerDoc["paths"] as JObject;
                
                if (paths == null)
                {
                    Console.WriteLine("No 'paths' section found in Swagger JSON");
                    return endpoints;
                }

                foreach (var pathProperty in paths.Properties())
                {
                    var path = pathProperty.Name;
                    var pathItem = pathProperty.Value as JObject;
                    
                    if (pathItem == null) continue;

                    // Process each HTTP method for this path
                    foreach (var methodProperty in pathItem.Properties())
                    {
                        var httpMethod = methodProperty.Name.ToLower();
                        
                        // Skip non-HTTP methods
                        if (!IsValidHttpMethod(httpMethod)) continue;
                        
                        var operation = methodProperty.Value as JObject;
                        if (operation == null) continue;

                        var endpoint = ParseOperation(path, httpMethod, operation, swaggerDoc);
                        if (endpoint != null)
                        {
                            endpoints.Add(endpoint);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error parsing Swagger JSON: {ex.Message}");
                throw;
            }

            return endpoints;
        }

        private bool IsValidHttpMethod(string method)
        {
            var validMethods = new[] { "get", "post", "put", "delete", "patch", "head", "options" };
            return validMethods.Contains(method.ToLower());
        }

        private SwaggerEndpoint ParseOperation(string path, string httpMethod, JObject operation, JObject swaggerDoc)
        {
            var endpoint = new SwaggerEndpoint
            {
                Path = path,
                HttpMethod = httpMethod.ToUpper(),
                OperationId = operation["operationId"]?.ToString() ?? GenerateOperationId(httpMethod, path),
                Summary = operation["summary"]?.ToString() ?? "",
                Description = operation["description"]?.ToString() ?? "",
                Tags = operation["tags"]?.ToObject<List<string>>() ?? new List<string>(),
                Consumes = operation["consumes"]?.ToObject<List<string>>() ?? new List<string>(),
                Produces = operation["produces"]?.ToObject<List<string>>() ?? new List<string>()
            };

            // Parse parameters
            var parameters = operation["parameters"] as JArray;
            if (parameters != null)
            {
                foreach (var param in parameters)
                {
                    var parameter = ParseParameter(param as JObject, swaggerDoc);
                    if (parameter != null)
                    {
                        endpoint.Parameters.Add(parameter);
                    }
                }
            }

            // Parse responses
            var responses = operation["responses"] as JObject;
            if (responses != null)
            {
                endpoint.ResponseType = ParseResponseType(responses, swaggerDoc);
            }

            return endpoint;
        }

        private EndpointParameter ParseParameter(JObject paramObj, JObject swaggerDoc)
        {
            if (paramObj == null) return null;

            var parameter = new EndpointParameter
            {
                Name = paramObj["name"]?.ToString() ?? "",
                Location = paramObj["in"]?.ToString() ?? "",
                IsRequired = paramObj["required"]?.ToObject<bool>() ?? false,
                Description = paramObj["description"]?.ToString() ?? ""
            };

            // Determine parameter type
            var type = paramObj["type"]?.ToString();
            var format = paramObj["format"]?.ToString();
            
            if (!string.IsNullOrEmpty(type))
            {
                parameter.Type = ConvertSwaggerTypeToCSharp(type, format);
            }
            else if (paramObj["schema"] != null)
            {
                // Handle complex types with schema
                parameter.Type = ParseSchemaType(paramObj["schema"] as JObject, swaggerDoc);
            }
            else
            {
                parameter.Type = "object";
            }

            return parameter;
        }

        private ResponseType ParseResponseType(JObject responses, JObject swaggerDoc)
        {
            var responseType = new ResponseType();
            
            // Look for successful response (200, 201, etc.)
            var successResponse = responses.Properties()
                .FirstOrDefault(p => p.Name.StartsWith("2"))?.Value as JObject;
            
            if (successResponse == null)
            {
                responseType.Type = "void";
                return responseType;
            }

            responseType.Description = successResponse["description"]?.ToString() ?? "";
            
            var schema = successResponse["schema"] as JObject;
            if (schema != null)
            {
                responseType.Type = ParseSchemaType(schema, swaggerDoc);
                
                // Check if it's an array
                if (schema["type"]?.ToString() == "array")
                {
                    responseType.IsArray = true;
                    var items = schema["items"] as JObject;
                    if (items != null)
                    {
                        responseType.Type = ParseSchemaType(items, swaggerDoc);
                    }
                }
            }
            else
            {
                responseType.Type = "void";
            }

            return responseType;
        }

        private string ParseSchemaType(JObject schema, JObject swaggerDoc)
        {
            if (schema == null) return "object";

            var type = schema["type"]?.ToString();
            var format = schema["format"]?.ToString();
            var reference = schema["$ref"]?.ToString();

            if (!string.IsNullOrEmpty(reference))
            {
                // Handle $ref to definitions
                var refName = reference.Split('/').Last();
                return refName;
            }

            if (!string.IsNullOrEmpty(type))
            {
                if (type == "array")
                {
                    var items = schema["items"] as JObject;
                    if (items != null)
                    {
                        var itemType = ParseSchemaType(items, swaggerDoc);
                        return $"List<{itemType}>";
                    }
                    return "List<object>";
                }
                
                return ConvertSwaggerTypeToCSharp(type, format);
            }

            return "object";
        }

        private string ConvertSwaggerTypeToCSharp(string swaggerType, string format = null)
        {
            return swaggerType.ToLower() switch
            {
                "integer" when format == "int64" => "long",
                "integer" => "int",
                "number" when format == "float" => "float",
                "number" when format == "double" => "double",
                "number" => "decimal",
                "string" when format == "date-time" => "DateTime",
                "string" when format == "date" => "DateTime",
                "string" when format == "uuid" => "Guid",
                "string" => "string",
                "boolean" => "bool",
                "array" => "List<object>",
                "object" => "object",
                _ => "object"
            };
        }

        private string GenerateOperationId(string httpMethod, string path)
        {
            // Generate a reasonable operation ID from method and path
            var pathParts = path.Split('/', StringSplitOptions.RemoveEmptyEntries)
                .Select(p => p.Replace("{", "").Replace("}", ""))
                .Select(p => char.ToUpper(p[0]) + p[1..]);
            
            return httpMethod.ToLower() + string.Join("", pathParts);
        }

        public void Dispose()
        {
            _httpClient?.Dispose();
        }
    }
}
