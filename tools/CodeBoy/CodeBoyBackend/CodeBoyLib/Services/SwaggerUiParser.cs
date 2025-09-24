using HtmlAgilityPack;
using CodeBoyLib.Models;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace CodeBoyLib.Services
{
    public class SwaggerUiParser
    {
        private readonly HttpClient _httpClient;

        public SwaggerUiParser()
        {
            _httpClient = new HttpClient();
        }

        public async Task<SwaggerApiInfo> Parse(string swaggerUrl)
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

        private SwaggerApiInfo ParseSwaggerJson(string jsonContent)
        {
            var apiInfo = new SwaggerApiInfo();
            
            try
            {
                var swaggerDoc = JObject.Parse(jsonContent);
                
                // Parse API info
                var info = swaggerDoc["info"] as JObject;
                if (info != null)
                {
                    apiInfo.Title = info["title"]?.ToString() ?? "";
                    apiInfo.Version = info["version"]?.ToString() ?? "";
                    apiInfo.Description = info["description"]?.ToString() ?? "";
                }

                // Parse base URL
                var host = swaggerDoc["host"]?.ToString();
                var basePath = swaggerDoc["basePath"]?.ToString();
                var schemes = swaggerDoc["schemes"] as JArray;
                var scheme = schemes?.FirstOrDefault()?.ToString() ?? "https";
                
                if (!string.IsNullOrEmpty(host))
                {
                    apiInfo.BaseUrl = $"{scheme}://{host}{basePath ?? ""}";
                }

                // Parse definitions (Swagger 2.0)
                var definitions = swaggerDoc["definitions"] as JObject;
                if (definitions != null)
                {
                    foreach (var definition in definitions.Properties())
                    {
                        var className = definition.Name;
                        var classSchema = definition.Value as JObject;
                        
                        if (classSchema != null)
                        {
                            var classDef = ParseClassDefinition(className, classSchema, swaggerDoc);
                            apiInfo.ClassDefinitions[className] = classDef;
                        }
                    }
                }

                // Parse components/schemas (OpenAPI 3.0)
                var components = swaggerDoc["components"] as JObject;
                if (components != null)
                {
                    var schemas = components["schemas"] as JObject;
                    if (schemas != null)
                    {
                        foreach (var schema in schemas.Properties())
                        {
                            var className = schema.Name;
                            var classSchema = schema.Value as JObject;
                            
                            if (classSchema != null)
                            {
                                var classDef = ParseClassDefinition(className, classSchema, swaggerDoc);
                                apiInfo.ClassDefinitions[className] = classDef;
                            }
                        }
                    }
                }

                // Parse paths (endpoints)
                var paths = swaggerDoc["paths"] as JObject;
                if (paths == null)
                {
                    Console.WriteLine("No 'paths' section found in Swagger JSON");
                    return apiInfo;
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

                        var endpoint = ParseOperation(path, httpMethod, operation, swaggerDoc, apiInfo);
                        if (endpoint != null)
                        {
                            apiInfo.Endpoints.Add(endpoint);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error parsing Swagger JSON: {ex.Message}");
                throw;
            }

            // Post-process: Convert numeric enum references to int type
            ConvertNumericEnumReferencesToInt(apiInfo);

            return apiInfo;
        }

        private void ConvertNumericEnumReferencesToInt(SwaggerApiInfo apiInfo)
        {
            // Get all numeric enum names
            var numericEnumNames = apiInfo.ClassDefinitions
                .Where(kvp => kvp.Value.IsNumericEnum)
                .Select(kvp => kvp.Key)
                .ToHashSet();

            if (numericEnumNames.Count == 0) return;

            // Update properties in all class definitions
            foreach (var classDef in apiInfo.ClassDefinitions.Values)
            {
                if (classDef.IsNumericEnum) continue; // Skip the numeric enums themselves

                foreach (var property in classDef.Properties)
                {
                    // Check direct type references
                    if (numericEnumNames.Contains(property.Type))
                    {
                        property.Type = "int";
                    }
                    // Check List<EnumType> references
                    else if (property.Type.StartsWith("List<") && property.Type.EndsWith(">"))
                    {
                        var innerType = property.Type.Substring(5, property.Type.Length - 6);
                        if (numericEnumNames.Contains(innerType))
                        {
                            property.Type = "List<int>";
                        }
                    }
                }
            }

            // Update parameter types in endpoints
            foreach (var endpoint in apiInfo.Endpoints)
            {
                foreach (var parameter in endpoint.Parameters)
                {
                    if (numericEnumNames.Contains(parameter.Type))
                    {
                        parameter.Type = "int";
                    }
                    else if (parameter.Type.StartsWith("List<") && parameter.Type.EndsWith(">"))
                    {
                        var innerType = parameter.Type.Substring(5, parameter.Type.Length - 6);
                        if (numericEnumNames.Contains(innerType))
                        {
                            parameter.Type = "List<int>";
                        }
                    }
                }

                // Update response types
                if (numericEnumNames.Contains(endpoint.ResponseType.Type))
                {
                    endpoint.ResponseType.Type = "int";
                }
                else if (endpoint.ResponseType.Type.StartsWith("List<") && endpoint.ResponseType.Type.EndsWith(">"))
                {
                    var innerType = endpoint.ResponseType.Type.Substring(5, endpoint.ResponseType.Type.Length - 6);
                    if (numericEnumNames.Contains(innerType))
                    {
                        endpoint.ResponseType.Type = "List<int>";
                    }
                }
            }
        }

        private bool IsNumericOnlyEnum(JArray enumValues)
        {
            if (enumValues == null || enumValues.Count == 0)
                return false;
                
            foreach (var value in enumValues)
            {
                // Check if value is numeric (integer or decimal)
                if (value.Type != JTokenType.Integer && value.Type != JTokenType.Float)
                {
                    return false;
                }
            }
            return true;
        }

        private ClassDefinition ParseClassDefinition(string className, JObject classSchema, JObject swaggerDoc)
        {
            var classDef = new ClassDefinition
            {
                Name = className,
                Description = classSchema["description"]?.ToString() ?? ""
            };

            var type = classSchema["type"]?.ToString();
            classDef.Type = type ?? "object";

            // Handle enums
            var enumValues = classSchema["enum"] as JArray;
            if (enumValues != null)
            {
                // Check if enum contains only numeric values
                bool isNumericOnlyEnum = IsNumericOnlyEnum(enumValues);
                
                if (!isNumericOnlyEnum)
                {
                    classDef.IsEnum = true;
                    foreach (var enumValue in enumValues)
                    {
                        classDef.EnumValues.Add(enumValue.ToString());
                    }
                    return classDef;
                }
                else
                {
                    // For numeric-only enums, treat as integer type and don't generate enum
                    // This class definition will be skipped by the code generator
                    classDef.IsEnum = false;
                    classDef.Type = "int"; // Mark as integer type
                    classDef.IsNumericEnum = true; // Flag to skip this class
                    return classDef;
                }
            }

            // Parse required properties
            var required = classSchema["required"] as JArray;
            if (required != null)
            {
                foreach (var req in required)
                {
                    classDef.RequiredProperties.Add(req.ToString());
                }
            }

            // Parse properties
            var properties = classSchema["properties"] as JObject;
            if (properties != null)
            {
                foreach (var property in properties.Properties())
                {
                    var propName = property.Name;
                    var propSchema = property.Value as JObject;
                    
                    if (propSchema != null)
                    {
                        var classProp = ParseClassProperty(propName, propSchema, classDef.RequiredProperties, swaggerDoc);
                        classDef.Properties.Add(classProp);
                    }
                }
            }

            return classDef;
        }

        private ClassProperty ParseClassProperty(string propName, JObject propSchema, List<string> requiredProperties, JObject swaggerDoc)
        {
            var classProp = new ClassProperty
            {
                Name = propName,
                Description = propSchema["description"]?.ToString() ?? "",
                IsRequired = requiredProperties.Contains(propName),
                Format = propSchema["format"]?.ToString() ?? ""
            };

            var type = propSchema["type"]?.ToString();
            var format = propSchema["format"]?.ToString();
            var reference = propSchema["$ref"]?.ToString();

            if (!string.IsNullOrEmpty(reference))
            {
                // Handle $ref to definitions
                var refName = reference.Split('/').Last();
                classProp.Type = refName;
            }
            else if (!string.IsNullOrEmpty(type))
            {
                if (type == "array")
                {
                    var items = propSchema["items"] as JObject;
                    if (items != null)
                    {
                        var itemType = ParseSchemaType(items, swaggerDoc);
                        classProp.Type = $"List<{itemType}>";
                    }
                    else
                    {
                        classProp.Type = "List<object>";
                    }
                }
                else
                {
                    classProp.Type = ConvertSwaggerTypeToCSharp(type, format);
                }
            }
            else
            {
                classProp.Type = "object";
            }

            // Handle nullable types - check explicit nullable flag first
            var isExplicitlyNullable = propSchema["nullable"]?.ToObject<bool>() ?? false;
            
            if (isExplicitlyNullable)
            {
                classProp.IsNullable = true;
                
                // For value types (primitives), add "?" to make them nullable
                if (IsPrimitiveType(classProp.Type) && !classProp.Type.EndsWith("?"))
                {
                    classProp.Type += "?";
                }
                // For reference types including List<T>, add "?" to make them nullable
                else if (!classProp.Type.EndsWith("?"))
                {
                    classProp.Type += "?";
                }
            }

            // Parse default value
            if (propSchema["default"] != null)
            {
                classProp.DefaultValue = propSchema["default"];
            }

            return classProp;
        }

        private bool IsPrimitiveType(string type)
        {
            var primitiveTypes = new HashSet<string>
            {
                "int", "long", "float", "double", "decimal", "bool", "DateTime", "Guid"
            };
            return primitiveTypes.Contains(type);
        }

        private bool IsValidHttpMethod(string method)
        {
            var validMethods = new[] { "get", "post", "put", "delete", "patch", "head", "options" };
            return validMethods.Contains(method.ToLower());
        }

        private SwaggerEndpoint ParseOperation(string path, string httpMethod, JObject operation, JObject swaggerDoc, SwaggerApiInfo apiInfo)
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

            // Parse parameters (Swagger 2.0 style)
            var parameters = operation["parameters"] as JArray;
            if (parameters != null)
            {
                foreach (var param in parameters)
                {
                    var parameter = ParseParameter(param as JObject, swaggerDoc, endpoint.OperationId, apiInfo);
                    if (parameter != null)
                    {
                        endpoint.Parameters.Add(parameter);
                    }
                }
            }

            // Parse requestBody (OpenAPI 3.0 style)
            var requestBody = operation["requestBody"] as JObject;
            if (requestBody != null)
            {
                var bodyParameter = ParseRequestBody(requestBody, swaggerDoc, endpoint.OperationId, apiInfo);
                if (bodyParameter != null)
                {
                    endpoint.Parameters.Add(bodyParameter);
                }
            }

            // Parse responses
            var responses = operation["responses"] as JObject;
            if (responses != null)
            {
                endpoint.ResponseType = ParseResponseType(responses, swaggerDoc, endpoint.OperationId, apiInfo);
            }

            return endpoint;
        }

        private EndpointParameter ParseRequestBody(JObject requestBody, JObject swaggerDoc, string operationId, SwaggerApiInfo apiInfo)
        {
            if (requestBody == null) return null;

            var content = requestBody["content"] as JObject;
            if (content == null) return null;

            // Try to find JSON content type
            JObject jsonContent = null;
            foreach (var contentType in new[] { "application/json", "application/json-patch+json", "text/json", "application/*+json" })
            {
                if (content[contentType] != null)
                {
                    jsonContent = content[contentType] as JObject;
                    break;
                }
            }

            if (jsonContent == null) return null;

            var schema = jsonContent["schema"] as JObject;
            if (schema == null) return null;

            var parameter = new EndpointParameter
            {
                Name = "body",
                Location = "body",
                IsRequired = true, // requestBody is typically required
                Description = requestBody["description"]?.ToString() ?? "Request body"
            };

            // Handle schema reference or inline schema
            var reference = schema["$ref"]?.ToString();
            if (!string.IsNullOrEmpty(reference))
            {
                // Handle $ref to components/schemas
                var refName = reference.Split('/').Last();
                parameter.Type = refName;
                
                // Ensure the referenced schema is parsed as a class definition
                if (!apiInfo.ClassDefinitions.ContainsKey(refName))
                {
                    var referencedSchema = GetReferencedSchema(reference, swaggerDoc);
                    if (referencedSchema != null)
                    {
                        var classDef = ParseClassDefinition(refName, referencedSchema, swaggerDoc);
                        apiInfo.ClassDefinitions[refName] = classDef;
                    }
                }
            }
            else if (schema["type"]?.ToString() == "array")
            {
                // Handle array type
                var items = schema["items"] as JObject;
                if (items != null)
                {
                    var itemReference = items["$ref"]?.ToString();
                    if (!string.IsNullOrEmpty(itemReference))
                    {
                        var itemRefName = itemReference.Split('/').Last();
                        parameter.Type = $"List<{itemRefName}>";
                        
                        // Ensure the referenced schema is parsed
                        if (!apiInfo.ClassDefinitions.ContainsKey(itemRefName))
                        {
                            var referencedSchema = GetReferencedSchema(itemReference, swaggerDoc);
                            if (referencedSchema != null)
                            {
                                var classDef = ParseClassDefinition(itemRefName, referencedSchema, swaggerDoc);
                                apiInfo.ClassDefinitions[itemRefName] = classDef;
                            }
                        }
                    }
                    else
                    {
                        parameter.Type = ParseSchemaType(items, swaggerDoc);
                    }
                }
                else
                {
                    parameter.Type = "List<object>";
                }
            }
            else
            {
                // Handle inline schema
                var dtoClassName = $"{operationId}Request";
                parameter.Type = dtoClassName;
                
                var classDef = ParseInlineSchemaToClassDefinition(dtoClassName, schema, swaggerDoc);
                if (classDef != null && !apiInfo.ClassDefinitions.ContainsKey(dtoClassName))
                {
                    apiInfo.ClassDefinitions[dtoClassName] = classDef;
                }
            }

            return parameter;
        }

        private JObject GetReferencedSchema(string reference, JObject swaggerDoc)
        {
            // Handle both Swagger 2.0 (#/definitions/) and OpenAPI 3.0 (#/components/schemas/) references
            var pathParts = reference.TrimStart('#').TrimStart('/').Split('/');
            
            JObject current = swaggerDoc;
            foreach (var part in pathParts)
            {
                current = current[part] as JObject;
                if (current == null) return null;
            }
            
            return current;
        }

        private EndpointParameter ParseParameter(JObject paramObj, JObject swaggerDoc, string operationId, SwaggerApiInfo apiInfo)
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
                var schema = paramObj["schema"] as JObject;
                parameter.Type = ParseSchemaType(schema, swaggerDoc);
                
                // For body parameters with complex inline schemas, generate DTO class
                if (parameter.Location == "body" && schema != null)
                {
                    var reference = schema["$ref"]?.ToString();
                    if (string.IsNullOrEmpty(reference))
                    {
                        // This is an inline schema, generate a DTO class for it
                        var dtoClassName = $"{operationId}Request";
                        parameter.Type = dtoClassName;
                        
                        // Generate the DTO class and add it to class definitions
                        var classDef = ParseInlineSchemaToClassDefinition(dtoClassName, schema, swaggerDoc);
                        if (classDef != null && !apiInfo.ClassDefinitions.ContainsKey(dtoClassName))
                        {
                            apiInfo.ClassDefinitions[dtoClassName] = classDef;
                        }
                    }
                }
            }
            else
            {
                parameter.Type = "object";
            }

            return parameter;
        }

        private ClassDefinition ParseInlineSchemaToClassDefinition(string className, JObject schema, JObject swaggerDoc)
        {
            if (schema == null) return null;
            
            return ParseClassDefinition(className, schema, swaggerDoc);
        }

        private JObject GetResponseSchema(JObject successResponse)
        {
            // Try OpenAPI 3.0 content format first
            var content = successResponse["content"] as JObject;
            if (content != null)
            {
                // Try to find JSON content type
                foreach (var contentType in new[] { "application/json", "text/json", "application/*+json", "text/plain" })
                {
                    var mediaType = content[contentType] as JObject;
                    if (mediaType != null)
                    {
                        var schema = mediaType["schema"] as JObject;
                        if (schema != null)
                        {
                            return schema;
                        }
                    }
                }
            }
            
            // Fallback to Swagger 2.0 format
            return successResponse["schema"] as JObject;
        }

        private ResponseType ParseResponseType(JObject responses, JObject swaggerDoc, string operationId, SwaggerApiInfo apiInfo)
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
            
            // Try to get schema from OpenAPI 3.0 content format first
            var schema = GetResponseSchema(successResponse);
            if (schema != null)
            {
                var reference = schema["$ref"]?.ToString();
                
                if (string.IsNullOrEmpty(reference))
                {
                    // This is an inline schema, generate a DTO class for it if it's complex
                    if (schema["type"]?.ToString() == "object" && schema["properties"] != null)
                    {
                        var dtoClassName = $"{operationId}Response";
                        responseType.Type = dtoClassName;
                        
                        // Generate the DTO class and add it to class definitions
                        var classDef = ParseInlineSchemaToClassDefinition(dtoClassName, schema, swaggerDoc);
                        if (classDef != null && !apiInfo.ClassDefinitions.ContainsKey(dtoClassName))
                        {
                            apiInfo.ClassDefinitions[dtoClassName] = classDef;
                        }
                    }
                    else
                    {
                        responseType.Type = ParseSchemaType(schema, swaggerDoc);
                    }
                }
                else
                {
                    responseType.Type = ParseSchemaType(schema, swaggerDoc);
                }
                
                // Check if it's an array
                if (schema["type"]?.ToString() == "array")
                {
                    responseType.IsArray = true;
                    var items = schema["items"] as JObject;
                    if (items != null)
                    {
                        var itemReference = items["$ref"]?.ToString();
                        if (string.IsNullOrEmpty(itemReference) && items["type"]?.ToString() == "object" && items["properties"] != null)
                        {
                            // Generate DTO for array item
                            var itemDtoClassName = $"{operationId}ResponseItem";
                            responseType.Type = itemDtoClassName;
                            
                            var itemClassDef = ParseInlineSchemaToClassDefinition(itemDtoClassName, items, swaggerDoc);
                            if (itemClassDef != null && !apiInfo.ClassDefinitions.ContainsKey(itemDtoClassName))
                            {
                                apiInfo.ClassDefinitions[itemDtoClassName] = itemClassDef;
                            }
                        }
                        else
                        {
                            responseType.Type = ParseSchemaType(items, swaggerDoc);
                        }
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
