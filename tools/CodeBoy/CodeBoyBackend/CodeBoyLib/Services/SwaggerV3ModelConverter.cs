using CodeBoyLib.Models;
using CodeBoyLib.Models.OpenApiV3;
using System;
using System.Collections.Generic;
using System.Linq;

namespace CodeBoyLib.Services
{
    /// <summary>
    /// 專門負責將 OpenAPI 3.0 模型轉換為 SwaggerApiInfo
    /// </summary>
    public class SwaggerV3ModelConverter
    {
        /// <summary>
        /// 將 OpenAPI 3.0 文檔轉換為 SwaggerApiInfo
        /// </summary>
        /// <param name="openApiDoc">OpenAPI 3.0 文檔</param>
        /// <returns>轉換後的 API 資訊</returns>
        public SwaggerApiInfo Convert(OpenApiDocument openApiDoc)
        {
            if (openApiDoc == null)
                throw new ArgumentNullException(nameof(openApiDoc));

            var apiInfo = new SwaggerApiInfo
            {
                Title = openApiDoc.Info?.Title ?? "",
                Version = openApiDoc.Info?.Version ?? "",
                Description = openApiDoc.Info?.Description ?? ""
            };

            // 建構 base URL
            BuildBaseUrl(openApiDoc, apiInfo);

            // 轉換 components/schemas 為 ClassDefinitions
            ConvertSchemas(openApiDoc, apiInfo);

            // 轉換 paths 為 Endpoints
            ConvertPaths(openApiDoc, apiInfo);

            return apiInfo;
        }

        /// <summary>
        /// 建構 Base URL（取第一個 server）
        /// </summary>
        private void BuildBaseUrl(OpenApiDocument openApiDoc, SwaggerApiInfo apiInfo)
        {
            var firstServer = openApiDoc.Servers?.FirstOrDefault();
            if (firstServer != null)
            {
                apiInfo.BaseUrl = firstServer.Url;
            }
        }

        /// <summary>
        /// 轉換 components/schemas 為 ClassDefinitions
        /// </summary>
        private void ConvertSchemas(OpenApiDocument openApiDoc, SwaggerApiInfo apiInfo)
        {
            if (openApiDoc.Components?.Schemas == null) return;

            foreach (var schema in openApiDoc.Components.Schemas)
            {
                var classDefinition = ConvertSchemaToClassDefinition(schema.Key, schema.Value, openApiDoc.Components.Schemas);
                if (classDefinition != null)
                {
                    apiInfo.ClassDefinitions[schema.Key] = classDefinition;
                }
            }
        }

        /// <summary>
        /// 轉換 paths 為 Endpoints
        /// </summary>
        private void ConvertPaths(OpenApiDocument openApiDoc, SwaggerApiInfo apiInfo)
        {
            if (openApiDoc.Paths == null) return;

            foreach (var path in openApiDoc.Paths)
            {
                var pathEndpoints = ConvertPathItemToEndpoints(path.Key, path.Value, openApiDoc, apiInfo);
                apiInfo.Endpoints.AddRange(pathEndpoints);
            }
        }

        /// <summary>
        /// 將 Schema 轉換為 ClassDefinition
        /// </summary>
        private ClassDefinition? ConvertSchemaToClassDefinition(string className, Schema schema, Dictionary<string, Schema> allSchemas)
        {
            if (schema == null) return null;

            var classDefinition = new ClassDefinition
            {
                Name = className,
                Description = schema.Description ?? ""
            };

            // 處理枚舉
            if (schema.Enum != null && schema.Enum.Count > 0)
            {
                classDefinition.IsEnum = true;
                classDefinition.EnumValues = schema.Enum.Select(e => e?.ToString() ?? "").ToList();

                // 檢查是否為純數字枚舉
                classDefinition.IsNumericEnum = schema.Enum.All(e =>
                    e != null && (e is int || e is long || e is decimal || e is double || e is float));

                return classDefinition;
            }

            // 處理物件屬性
            if (schema.Properties != null)
            {
                var requiredProperties = schema.Required ?? new List<string>();

                foreach (var property in schema.Properties)
                {
                    var classProp = ConvertSchemaToClassProperty(property.Key, property.Value, requiredProperties, allSchemas);
                    if (classProp != null)
                    {
                        classDefinition.Properties.Add(classProp);
                    }
                }
            }

            return classDefinition;
        }

        /// <summary>
        /// 將 Schema 轉換為 ClassProperty
        /// </summary>
        private ClassProperty? ConvertSchemaToClassProperty(string propName, Schema propSchema, List<string> requiredProperties, Dictionary<string, Schema> allSchemas)
        {
            if (propSchema == null) return null;

            var classProp = new ClassProperty
            {
                Name = propName,
                Description = propSchema.Description ?? "",
                IsRequired = requiredProperties.Contains(propName)
            };

            // 處理引用類型
            if (!string.IsNullOrEmpty(propSchema.Ref))
            {
                var refTypeName = ExtractTypeNameFromRef(propSchema.Ref);
                
                // Check if the referenced type is a numeric enum, if so use int instead
                if (allSchemas.TryGetValue(refTypeName, out var refSchema) && 
                    refSchema.Enum != null && refSchema.Enum.Count > 0 &&
                    refSchema.Enum.All(e => e != null && (e is int || e is long || e is decimal || e is double || e is float)))
                {
                    classProp.Type = "int";
                }
                else
                {
                    classProp.Type = refTypeName;
                }
                return classProp;
            }

            // 處理基本類型
            var schemaType = propSchema.Type?.ToLower();
            switch (schemaType)
            {
                case "string":
                    if (propSchema.Enum != null && propSchema.Enum.Count > 0)
                    {
                        // 這是一個枚舉字符串
                        classProp.Type = $"{propName}Enum"; // 可能需要產生內聯枚舉
                    }
                    else
                    {
                        classProp.Type = "string";
                    }
                    break;
                case "integer":
                    classProp.Type = propSchema.Format == "int64" ? "long" : "int";
                    break;
                case "number":
                    classProp.Type = propSchema.Format == "float" ? "float" : "decimal";
                    break;
                case "boolean":
                    classProp.Type = "bool";
                    break;
                case "array":
                    if (propSchema.Items != null)
                    {
                        var itemType = GetSchemaType(propSchema.Items, allSchemas);
                        classProp.Type = $"List<{itemType}>";
                    }
                    else
                    {
                        classProp.Type = "List<object>";
                    }
                    break;
                case "object":
                    classProp.Type = "object"; // 可能需要產生內聯類別
                    break;
                default:
                    classProp.Type = "object";
                    break;
            }

            return classProp;
        }

        /// <summary>
        /// 將 PathItem 轉換為 SwaggerEndpoint 列表
        /// </summary>
        private List<SwaggerEndpoint> ConvertPathItemToEndpoints(string path, PathItem pathItem, OpenApiDocument openApiDoc, SwaggerApiInfo apiInfo)
        {
            var endpoints = new List<SwaggerEndpoint>();

            var operations = new Dictionary<string, Operation?>
            {
                { "GET", pathItem.Get },
                { "POST", pathItem.Post },
                { "PUT", pathItem.Put },
                { "DELETE", pathItem.Delete },
                { "OPTIONS", pathItem.Options },
                { "HEAD", pathItem.Head },
                { "PATCH", pathItem.Patch },
                { "TRACE", pathItem.Trace }
            };

            foreach (var operation in operations)
            {
                if (operation.Value != null)
                {
                    var endpoint = ConvertOperationToEndpoint(path, operation.Key, operation.Value, openApiDoc, apiInfo);
                    if (endpoint != null)
                    {
                        endpoints.Add(endpoint);
                    }
                }
            }

            return endpoints;
        }

        /// <summary>
        /// 將 Operation 轉換為 SwaggerEndpoint
        /// </summary>
        private SwaggerEndpoint? ConvertOperationToEndpoint(string path, string httpMethod, Operation operation, OpenApiDocument openApiDoc, SwaggerApiInfo apiInfo)
        {
            var endpoint = new SwaggerEndpoint
            {
                Path = path,
                HttpMethod = httpMethod.ToUpper(),
                OperationId = operation.OperationId ?? $"{httpMethod.ToLower()}{path.Replace("/", "").Replace("{", "").Replace("}", "")}",
                Summary = operation.Summary ?? "",
                Description = operation.Description ?? "",
                Tags = operation.Tags ?? new List<string>()
            };

            // 轉換參數
            if (operation.Parameters != null)
            {
                foreach (var param in operation.Parameters)
                {
                    var endpointParam = ConvertParameterToEndpointParameter(param, openApiDoc, endpoint.OperationId, apiInfo);
                    if (endpointParam != null)
                    {
                        endpoint.Parameters.Add(endpointParam);
                    }
                }
            }

            // 轉換 requestBody
            if (operation.RequestBody != null)
            {
                var requestBodyParam = ConvertRequestBodyToEndpointParameter(operation.RequestBody, openApiDoc, endpoint.OperationId, apiInfo);
                if (requestBodyParam != null)
                {
                    endpoint.Parameters.Add(requestBodyParam);
                }
            }

            // 轉換回應
            endpoint.ResponseType = ConvertResponsesToResponseType(operation.Responses, openApiDoc, endpoint.OperationId, apiInfo);

            return endpoint;
        }

        /// <summary>
        /// 將 Parameter 轉換為 EndpointParameter
        /// </summary>
        private EndpointParameter? ConvertParameterToEndpointParameter(Parameter param, OpenApiDocument openApiDoc, string operationId, SwaggerApiInfo apiInfo)
        {
            if (param == null) return null;

            var endpointParam = new EndpointParameter
            {
                Name = param.Name,
                Location = param.In,
                IsRequired = param.Required,
                Description = param.Description ?? ""
            };

            // 設定參數類型
            if (param.Schema != null)
            {
                endpointParam.Type = GetSchemaType(param.Schema, openApiDoc.Components?.Schemas ?? new Dictionary<string, Schema>());
            }
            else
            {
                endpointParam.Type = "object";
            }

            return endpointParam;
        }

        /// <summary>
        /// 將 RequestBody 轉換為 EndpointParameter
        /// </summary>
        private EndpointParameter? ConvertRequestBodyToEndpointParameter(RequestBody requestBody, OpenApiDocument openApiDoc, string operationId, SwaggerApiInfo apiInfo)
        {
            if (requestBody == null) return null;

            // 尋找 JSON content type
            var jsonContent = requestBody.Content.FirstOrDefault(c => c.Key.Contains("json"));
            if (jsonContent.Value?.Schema != null)
            {
                var endpointParam = new EndpointParameter
                {
                    Name = "body",
                    Location = "body",
                    IsRequired = requestBody.Required,
                    Description = requestBody.Description ?? ""
                };

                endpointParam.Type = GetSchemaType(jsonContent.Value.Schema, openApiDoc.Components?.Schemas ?? new Dictionary<string, Schema>());

                return endpointParam;
            }

            return null;
        }

        /// <summary>
        /// 將 Responses 轉換為 ResponseType
        /// </summary>
        private ResponseType ConvertResponsesToResponseType(Dictionary<string, Response> responses, OpenApiDocument openApiDoc, string operationId, SwaggerApiInfo apiInfo)
        {
            var responseType = new ResponseType();

            // 尋找成功回應（2xx）
            var successResponse = responses.FirstOrDefault(r => r.Key.StartsWith("2"));
            if (successResponse.Value != null)
            {
                responseType.Description = successResponse.Value.Description;

                if (successResponse.Value.Content != null)
                {
                    var jsonContent = successResponse.Value.Content.FirstOrDefault(c => c.Key.Contains("json"));
                    if (jsonContent.Value?.Schema != null)
                    {
                        var schema = jsonContent.Value.Schema;

                        // 處理陣列回應
                        if (schema.Type == "array" && schema.Items != null)
                        {
                            responseType.IsArray = true;
                            responseType.Type = GetSchemaType(schema.Items, openApiDoc.Components?.Schemas ?? new Dictionary<string, Schema>());
                        }
                        else
                        {
                            responseType.Type = GetSchemaType(schema, openApiDoc.Components?.Schemas ?? new Dictionary<string, Schema>());
                        }
                    }
                    else
                    {
                        responseType.Type = "void";
                    }
                }
                else
                {
                    responseType.Type = "void";
                }
            }
            else
            {
                // 沒有成功回應，檢查 default
                if (responses.ContainsKey("default"))
                {
                    responseType.Description = responses["default"].Description;
                    responseType.Type = "void";
                }
                else
                {
                    responseType.Type = "void";
                }
            }

            return responseType;
        }

        /// <summary>
        /// 獲取 Schema 的 C# 類型
        /// </summary>
        private string GetSchemaType(Schema schema, Dictionary<string, Schema> allSchemas)
        {
            if (schema == null) return "object";

            if (!string.IsNullOrEmpty(schema.Ref))
            {
                var refTypeName = ExtractTypeNameFromRef(schema.Ref);
                
                // Check if the referenced type is a numeric enum, if so use int instead
                if (allSchemas.TryGetValue(refTypeName, out var refSchema) && 
                    refSchema.Enum != null && refSchema.Enum.Count > 0 &&
                    refSchema.Enum.All(e => e != null && (e is int || e is long || e is decimal || e is double || e is float)))
                {
                    return "int";
                }
                
                return refTypeName;
            }

            var schemaType = schema.Type?.ToLower();
            return schemaType switch
            {
                "string" => "string",
                "integer" => schema.Format == "int64" ? "long" : "int",
                "number" => schema.Format == "float" ? "float" : "decimal",
                "boolean" => "bool",
                "array" => schema.Items != null ? $"List<{GetSchemaType(schema.Items, allSchemas)}>" : "List<object>",
                "object" => "object",
                _ => "object"
            };
        }

        /// <summary>
        /// 從 $ref 中提取類型名稱
        /// </summary>
        private string ExtractTypeNameFromRef(string reference)
        {
            if (string.IsNullOrEmpty(reference)) return "object";

            // 處理 OpenAPI 3.0: "#/components/schemas/User"
            var parts = reference.Split('/');
            return parts.LastOrDefault() ?? "object";
        }
    }
}
