using MakeSwaggerSDK.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MakeSwaggerSDK.Services
{
    public class SwaggerClientCodeGenerator
    {
        public string Generate(string sdkName, List<SwaggerEndpoint> endpoints)
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
            sb.AppendLine();

            // Add namespace
            sb.AppendLine($"namespace {sdkName}SDK");
            sb.AppendLine("{");

            // Generate DTO classes for complex response types
            var responseTypes = GetUniqueResponseTypes(endpoints);
            foreach (var responseType in responseTypes)
            {
                if (responseType != "void" && responseType != "string" && responseType != "int" && 
                    responseType != "bool" && responseType != "DateTime" && responseType != "decimal" &&
                    responseType != "long" && responseType != "float" && responseType != "double" &&
                    responseType != "Guid" && responseType != "object" && 
                    !responseType.StartsWith("List<") && !IsPrimitiveType(responseType))
                {
                    GenerateResponseClass(sb, responseType);
                }
            }

            // Generate the main client class
            GenerateClientClass(sb, sdkName, endpoints);

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

        private void GenerateResponseClass(StringBuilder sb, string className)
        {
            sb.AppendLine($"    /// <summary>");
            sb.AppendLine($"    /// Response model for {className}");
            sb.AppendLine($"    /// </summary>");
            sb.AppendLine($"    public class {className}");
            sb.AppendLine("    {");
            sb.AppendLine("        // TODO: Add properties based on actual API response structure");
            sb.AppendLine("        // You may need to manually define the properties for this class");
            sb.AppendLine("    }");
            sb.AppendLine();
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

            // Remove special characters and ensure it starts with a letter
            var sanitized = new StringBuilder();
            bool firstChar = true;

            foreach (char c in operationId)
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
                    // Replace special chars with underscore, but avoid consecutive underscores
                    if (sanitized.Length > 0 && sanitized[sanitized.Length - 1] != '_')
                    {
                        sanitized.Append('_');
                    }
                }
            }

            var result = sanitized.ToString().TrimEnd('_');
            return string.IsNullOrEmpty(result) ? "UnknownOperation" : result;
        }
    }
}
