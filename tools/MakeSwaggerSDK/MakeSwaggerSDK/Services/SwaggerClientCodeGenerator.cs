using MakeSwaggerSDK.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MakeSwaggerSDK.Services
{
    public class SwaggerClientCodeGenerator
    {
        public string Generate(string sdkName, SwaggerApiInfo apiInfo)
        {
            var sb = new StringBuilder();
            
            // Add using statements
            sb.AppendLine("using System;");
            sb.AppendLine("using System.Collections.Generic;");
            sb.AppendLine("using System.Linq;");
            sb.AppendLine("using System.Net.Http;");
            sb.AppendLine("using System.Text;");
            sb.AppendLine("using System.Threading.Tasks;");
            sb.AppendLine("using Microsoft.Extensions.Http;");
            sb.AppendLine("using Newtonsoft.Json;");
            sb.AppendLine("using System.ComponentModel.DataAnnotations;");
            sb.AppendLine();

            // Add namespace
            sb.AppendLine($"namespace {sdkName}SDK");
            sb.AppendLine("{");

            // Generate model classes from definitions
            foreach (var classDef in apiInfo.ClassDefinitions.Values)
            {
                GenerateModelClass(sb, classDef);
            }

            // Generate the main client class
            GenerateClientClass(sb, sdkName, apiInfo.Endpoints);

            sb.AppendLine("}");

            return sb.ToString();
        }

        private HashSet<string> GetUniqueResponseTypes(List<SwaggerEndpoint> endpoints)
        {
            var types = new HashSet<string>();
            foreach (var endpoint in endpoints)
            {
                if (!string.IsNullOrEmpty(endpoint.ResponseType.Type))
                {
                    types.Add(endpoint.ResponseType.Type);
                }
            }
            return types;
        }

        private bool IsPrimitiveType(string type)
        {
            var primitiveTypes = new HashSet<string>
            {
                "string", "int", "long", "float", "double", "decimal", "bool", "DateTime", "Guid", "object", "void"
            };
            return primitiveTypes.Contains(type) || type.StartsWith("List<");
        }

        private void GenerateModelClass(StringBuilder sb, ClassDefinition classDef)
        {
            sb.AppendLine($"    /// <summary>");
            sb.AppendLine($"    /// {classDef.Description}");
            sb.AppendLine($"    /// </summary>");
            
            if (classDef.IsEnum)
            {
                GenerateEnumClass(sb, classDef);
            }
            else
            {
                GenerateObjectClass(sb, classDef);
            }
        }

        private void GenerateEnumClass(StringBuilder sb, ClassDefinition classDef)
        {
            sb.AppendLine($"    public enum {classDef.Name}");
            sb.AppendLine("    {");
            
            for (int i = 0; i < classDef.EnumValues.Count; i++)
            {
                var enumValue = classDef.EnumValues[i];
                var sanitizedValue = SanitizeEnumValue(enumValue);
                
                if (i == classDef.EnumValues.Count - 1)
                {
                    sb.AppendLine($"        {sanitizedValue}");
                }
                else
                {
                    sb.AppendLine($"        {sanitizedValue},");
                }
            }
            
            sb.AppendLine("    }");
            sb.AppendLine();
        }

        private void GenerateObjectClass(StringBuilder sb, ClassDefinition classDef)
        {
            sb.AppendLine($"    public class {classDef.Name}");
            sb.AppendLine("    {");
            
            foreach (var property in classDef.Properties)
            {
                GenerateProperty(sb, property);
            }
            
            sb.AppendLine("    }");
            sb.AppendLine();
        }

        private void GenerateProperty(StringBuilder sb, ClassProperty property)
        {
            // Add XML documentation
            if (!string.IsNullOrEmpty(property.Description))
            {
                sb.AppendLine($"        /// <summary>");
                sb.AppendLine($"        /// {property.Description}");
                sb.AppendLine($"        /// </summary>");
            }

            // Add validation attributes
            if (property.IsRequired)
            {
                sb.AppendLine($"        [Required]");
            }

            // Add JSON property attribute
            sb.AppendLine($"        [JsonProperty(\"{property.Name}\")]");

            // Generate property declaration
            var defaultValue = GetDefaultValueString(property);
            if (!string.IsNullOrEmpty(defaultValue))
            {
                sb.AppendLine($"        public {property.Type} {PascalCase(property.Name)} {{ get; set; }} = {defaultValue};");
            }
            else
            {
                if (property.Type.EndsWith("?") || !property.IsRequired)
                {
                    sb.AppendLine($"        public {property.Type} {PascalCase(property.Name)} {{ get; set; }}");
                }
                else
                {
                    sb.AppendLine($"        public {property.Type} {PascalCase(property.Name)} {{ get; set; }} = default!;");
                }
            }
            
            sb.AppendLine();
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

        private void GenerateClientClass(StringBuilder sb, string sdkName, List<SwaggerEndpoint> endpoints)
        {
            var className = $"{sdkName}Client";
            
            sb.AppendLine($"    /// <summary>");
            sb.AppendLine($"    /// HTTP client for {sdkName} API");
            sb.AppendLine($"    /// </summary>");
            sb.AppendLine($"    public class {className}");
            sb.AppendLine("    {");
            sb.AppendLine("        private readonly HttpClient _httpClient;");
            sb.AppendLine("        private readonly string _baseUrl;");
            sb.AppendLine();

            // Constructor with IHttpClientFactory
            sb.AppendLine($"        /// <summary>");
            sb.AppendLine($"        /// Initializes a new instance of {className}");
            sb.AppendLine($"        /// </summary>");
            sb.AppendLine($"        /// <param name=\"httpClientFactory\">HTTP client factory</param>");
            sb.AppendLine($"        /// <param name=\"baseUrl\">Base URL for the API</param>");
            sb.AppendLine($"        public {className}(IHttpClientFactory httpClientFactory, string baseUrl)");
            sb.AppendLine("        {");
            sb.AppendLine("            _httpClient = httpClientFactory.CreateClient();");
            sb.AppendLine("            _baseUrl = baseUrl.TrimEnd('/');");
            sb.AppendLine("        }");
            sb.AppendLine();

            // Alternative constructor with HttpClient directly
            sb.AppendLine($"        /// <summary>");
            sb.AppendLine($"        /// Initializes a new instance of {className}");
            sb.AppendLine($"        /// </summary>");
            sb.AppendLine($"        /// <param name=\"httpClient\">HTTP client instance</param>");
            sb.AppendLine($"        /// <param name=\"baseUrl\">Base URL for the API</param>");
            sb.AppendLine($"        public {className}(HttpClient httpClient, string baseUrl)");
            sb.AppendLine("        {");
            sb.AppendLine("            _httpClient = httpClient ?? throw new ArgumentNullException(nameof(httpClient));");
            sb.AppendLine("            _baseUrl = baseUrl.TrimEnd('/');");
            sb.AppendLine("        }");
            sb.AppendLine();

            // Generate methods for each endpoint
            foreach (var endpoint in endpoints)
            {
                GenerateEndpointMethod(sb, endpoint);
            }

            // Helper methods
            GenerateHelperMethods(sb);

            sb.AppendLine("    }");
        }

        private void GenerateEndpointMethod(StringBuilder sb, SwaggerEndpoint endpoint)
        {
            var methodName = SanitizeMethodName(endpoint.OperationId);
            var returnType = GetReturnType(endpoint.ResponseType);
            
            sb.AppendLine($"        /// <summary>");
            sb.AppendLine($"        /// {endpoint.Summary}");
            if (!string.IsNullOrEmpty(endpoint.Description))
            {
                sb.AppendLine($"        /// {endpoint.Description}");
            }
            sb.AppendLine($"        /// </summary>");

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
                sb.AppendLine($"        /// <param name=\"{param.Name}\">{param.Description}</param>");
            }

            // Add body parameter if exists
            if (bodyParams.Any())
            {
                var bodyParam = bodyParams.First();
                methodParams.Add($"{bodyParam.Type} {bodyParam.Name}");
                sb.AppendLine($"        /// <param name=\"{bodyParam.Name}\">{bodyParam.Description}</param>");
            }

            // Add query parameters as optional
            foreach (var param in queryParams)
            {
                var paramType = param.IsRequired ? param.Type : $"{param.Type}?";
                var defaultValue = param.IsRequired ? "" : " = null";
                methodParams.Add($"{paramType} {param.Name}{defaultValue}");
                sb.AppendLine($"        /// <param name=\"{param.Name}\">{param.Description}</param>");
            }

            // Add header parameters as optional
            foreach (var param in headerParams)
            {
                var paramType = param.IsRequired ? param.Type : $"{param.Type}?";
                var defaultValue = param.IsRequired ? "" : " = null";
                methodParams.Add($"{paramType} {param.Name}{defaultValue}");
                sb.AppendLine($"        /// <param name=\"{param.Name}\">{param.Description}</param>");
            }

            sb.AppendLine($"        /// <returns>{returnType}</returns>");
            sb.AppendLine($"        public async {returnType} {methodName}({string.Join(", ", methodParams)})");
            sb.AppendLine("        {");

            // Build URL
            var urlBuilder = new StringBuilder();
            urlBuilder.Append($"            var url = $\"{endpoint.Path}\"");
            
            // Replace path parameters
            foreach (var param in pathParams)
            {
                urlBuilder.Replace($"{{{param.Name}}}", $"{{{param.Name}}}");
            }
            urlBuilder.Append(";");
            sb.AppendLine(urlBuilder.ToString());

            // Add query parameters
            if (queryParams.Any())
            {
                sb.AppendLine("            var queryParams = new List<string>();");
                foreach (var param in queryParams)
                {
                    if (param.IsRequired)
                    {
                        sb.AppendLine($"            queryParams.Add($\"{param.Name}={{Uri.EscapeDataString({param.Name}.ToString())}}\");");
                    }
                    else
                    {
                        sb.AppendLine($"            if ({param.Name} != null)");
                        sb.AppendLine($"                queryParams.Add($\"{param.Name}={{Uri.EscapeDataString({param.Name}.ToString())}}\");");
                    }
                }
                sb.AppendLine("            if (queryParams.Any())");
                sb.AppendLine("                url += \"?\" + string.Join(\"&\", queryParams);");
                sb.AppendLine();
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
            sb.AppendLine($"            var request = new HttpRequestMessage({httpMethod}, _baseUrl + url);");

            // Add headers
            foreach (var param in headerParams)
            {
                if (param.IsRequired)
                {
                    sb.AppendLine($"            request.Headers.Add(\"{param.Name}\", {param.Name}.ToString());");
                }
                else
                {
                    sb.AppendLine($"            if ({param.Name} != null)");
                    sb.AppendLine($"                request.Headers.Add(\"{param.Name}\", {param.Name}.ToString());");
                }
            }

            // Add body content for POST/PUT/PATCH
            if (bodyParams.Any() && (endpoint.HttpMethod.ToUpper() == "POST" || endpoint.HttpMethod.ToUpper() == "PUT" || endpoint.HttpMethod.ToUpper() == "PATCH"))
            {
                var bodyParam = bodyParams.First();
                sb.AppendLine($"            var jsonContent = JsonConvert.SerializeObject({bodyParam.Name});");
                sb.AppendLine("            request.Content = new StringContent(jsonContent, Encoding.UTF8, \"application/json\");");
            }

            sb.AppendLine();
            sb.AppendLine("            var response = await _httpClient.SendAsync(request);");
            sb.AppendLine("            response.EnsureSuccessStatusCode();");
            sb.AppendLine();

            // Handle response
            if (returnType == "Task")
            {
                sb.AppendLine("            // No return value expected");
            }
            else if (returnType == "Task<string>")
            {
                sb.AppendLine("            return await response.Content.ReadAsStringAsync();");
            }
            else
            {
                sb.AppendLine("            var responseContent = await response.Content.ReadAsStringAsync();");
                if (returnType.StartsWith("Task<List<"))
                {
                    var innerType = returnType.Substring(5, returnType.Length - 6); // Remove "Task<" and ">"
                    sb.AppendLine($"            return JsonConvert.DeserializeObject<{innerType}>(responseContent) ?? new {innerType}();");
                }
                else if (returnType.StartsWith("Task<"))
                {
                    var innerType = returnType.Substring(5, returnType.Length - 6); // Remove "Task<" and ">"
                    sb.AppendLine($"            return JsonConvert.DeserializeObject<{innerType}>(responseContent);");
                }
            }

            sb.AppendLine("        }");
            sb.AppendLine();
        }

        private void GenerateHelperMethods(StringBuilder sb)
        {
            sb.AppendLine("        /// <summary>");
            sb.AppendLine("        /// Dispose resources");
            sb.AppendLine("        /// </summary>");
            sb.AppendLine("        public void Dispose()");
            sb.AppendLine("        {");
            sb.AppendLine("            _httpClient?.Dispose();");
            sb.AppendLine("        }");
            sb.AppendLine();
        }

        private string GetReturnType(ResponseType responseType)
        {
            if (responseType == null || string.IsNullOrEmpty(responseType.Type) || responseType.Type == "void")
            {
                return "Task";
            }

            if (responseType.IsArray)
            {
                return $"Task<List<{responseType.Type}>>";
            }

            return $"Task<{responseType.Type}>";
        }

        private string SanitizeMethodName(string operationId)
        {
            if (string.IsNullOrEmpty(operationId))
            {
                return "UnknownOperation";
            }

            // Convert to PascalCase: all words capitalized
            var words = new List<string>();
            var currentWord = new StringBuilder();
            
            // Split into words based on special characters, underscores, and camelCase
            for (int i = 0; i < operationId.Length; i++)
            {
                char c = operationId[i];
                
                if (char.IsLetterOrDigit(c))
                {
                    // Check if this is the start of a new word (uppercase after lowercase)
                    if (i > 0 && char.IsUpper(c) && char.IsLower(operationId[i - 1]))
                    {
                        if (currentWord.Length > 0)
                        {
                            words.Add(currentWord.ToString());
                            currentWord.Clear();
                        }
                    }
                    currentWord.Append(c);
                }
                else
                {
                    // Non-alphanumeric character - end current word
                    if (currentWord.Length > 0)
                    {
                        words.Add(currentWord.ToString());
                        currentWord.Clear();
                    }
                }
            }
            
            // Add the last word if any
            if (currentWord.Length > 0)
            {
                words.Add(currentWord.ToString());
            }

            if (words.Count == 0)
            {
                return "UnknownOperation";
            }

            // Build PascalCase result
            var result = new StringBuilder();
            for (int i = 0; i < words.Count; i++)
            {
                var word = words[i].ToLower();
                // All words: capitalize first letter
                if (word.Length > 0)
                {
                    result.Append(char.ToUpper(word[0]));
                    if (word.Length > 1)
                    {
                        result.Append(word.Substring(1));
                    }
                }
            }

            var finalResult = result.ToString();
            
            // Ensure the result starts with a letter, not a digit
            if (char.IsDigit(finalResult[0]))
            {
                finalResult = "Operation" + finalResult;
            }

            return string.IsNullOrEmpty(finalResult) ? "UnknownOperation" : finalResult;
        }
    }
}
