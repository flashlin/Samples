using CodeBoyLib.Models;
using CodeBoyLib.Models.SwaggerV2;
using CodeBoyLib.Models.OpenApiV3;
using System;
using System.Collections.Generic;
using System.Linq;

namespace CodeBoyLib.Services
{
    /// <summary>
    /// 將強型別的 Swagger/OpenAPI 模型轉換為現有的 SwaggerApiInfo 結構
    /// </summary>
    public class SwaggerModelConverter
    {
        /// <summary>
        /// 將 Swagger 2.0 文檔轉換為 SwaggerApiInfo
        /// </summary>
        /// <param name="swaggerDoc">Swagger 2.0 文檔</param>
        /// <returns>轉換後的 API 資訊</returns>
        public SwaggerApiInfo ConvertSwaggerV2ToApiInfo(Models.SwaggerV2.SwaggerDocument swaggerDoc)
        {
            if (swaggerDoc == null)
                throw new ArgumentNullException(nameof(swaggerDoc));

            var apiInfo = new SwaggerApiInfo
            {
                Title = swaggerDoc.Info?.Title ?? "",
                Version = swaggerDoc.Info?.Version ?? "",
                Description = swaggerDoc.Info?.Description ?? ""
            };

            // 建構 base URL
            var scheme = swaggerDoc.Schemes?.FirstOrDefault() ?? "https";
            var host = swaggerDoc.Host ?? "";
            var basePath = swaggerDoc.BasePath?.TrimStart('/') ?? "";
            
            if (!string.IsNullOrEmpty(host))
            {
                apiInfo.BaseUrl = $"{scheme}://{host}";
                if (!string.IsNullOrEmpty(basePath))
                {
                    apiInfo.BaseUrl += "/" + basePath;
                }
            }

            // 轉換 definitions 為 ClassDefinitions
            if (swaggerDoc.Definitions != null)
            {
                foreach (var definition in swaggerDoc.Definitions)
                {
                    var classDefinition = ConvertSchemaToClassDefinition(definition.Key, definition.Value, swaggerDoc.Definitions);
                    if (classDefinition != null)
                    {
                        apiInfo.ClassDefinitions[definition.Key] = classDefinition;
                    }
                }
            }

            // 轉換 paths 為 Endpoints
            if (swaggerDoc.Paths != null)
            {
                foreach (var path in swaggerDoc.Paths)
                {
                    var pathEndpoints = ConvertPathItemToEndpoints(path.Key, path.Value, swaggerDoc, apiInfo);
                    apiInfo.Endpoints.AddRange(pathEndpoints);
                }
            }

            return apiInfo;
        }

        /// <summary>
        /// 將 OpenAPI 3.0 文檔轉換為 SwaggerApiInfo
        /// </summary>
        /// <param name="openApiDoc">OpenAPI 3.0 文檔</param>
        /// <returns>轉換後的 API 資訊</returns>
        public SwaggerApiInfo ConvertOpenApiV3ToApiInfo(Models.OpenApiV3.OpenApiDocument openApiDoc)
        {
            if (openApiDoc == null)
                throw new ArgumentNullException(nameof(openApiDoc));

            var apiInfo = new SwaggerApiInfo
            {
                Title = openApiDoc.Info?.Title ?? "",
                Version = openApiDoc.Info?.Version ?? "",
                Description = openApiDoc.Info?.Description ?? ""
            };

            // 建構 base URL（取第一個 server）
            var firstServer = openApiDoc.Servers?.FirstOrDefault();
            if (firstServer != null)
            {
                apiInfo.BaseUrl = firstServer.Url;
            }

            // 轉換 components/schemas 為 ClassDefinitions
            if (openApiDoc.Components?.Schemas != null)
            {
                foreach (var schema in openApiDoc.Components.Schemas)
                {
                    var classDefinition = ConvertOpenApiSchemaToClassDefinition(schema.Key, schema.Value, openApiDoc.Components.Schemas);
                    if (classDefinition != null)
                    {
                        apiInfo.ClassDefinitions[schema.Key] = classDefinition;
                    }
                }
            }

            // 轉換 paths 為 Endpoints
            if (openApiDoc.Paths != null)
            {
                foreach (var path in openApiDoc.Paths)
                {
                    var pathEndpoints = ConvertOpenApiPathItemToEndpoints(path.Key, path.Value, openApiDoc, apiInfo);
                    apiInfo.Endpoints.AddRange(pathEndpoints);
                }
            }

            return apiInfo;
        }

        /// <summary>
        /// 將 Swagger 2.0 Schema 轉換為 ClassDefinition
        /// </summary>
        private ClassDefinition? ConvertSchemaToClassDefinition(string className, Models.SwaggerV2.Schema schema, Dictionary<string, Models.SwaggerV2.Schema> allDefinitions)
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
                    var classProp = ConvertSchemaToClassProperty(property.Key, property.Value, requiredProperties, allDefinitions);
                    if (classProp != null)
                    {
                        classDefinition.Properties.Add(classProp);
                    }
                }
            }

            return classDefinition;
        }

        /// <summary>
        /// 將 OpenAPI 3.0 Schema 轉換為 ClassDefinition
        /// </summary>
        private ClassDefinition? ConvertOpenApiSchemaToClassDefinition(string className, Models.OpenApiV3.Schema schema, Dictionary<string, Models.OpenApiV3.Schema> allSchemas)
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
                    var classProp = ConvertOpenApiSchemaToClassProperty(property.Key, property.Value, requiredProperties, allSchemas);
                    if (classProp != null)
                    {
                        classDefinition.Properties.Add(classProp);
                    }
                }
            }

            return classDefinition;
        }

        /// <summary>
        /// 將 Swagger 2.0 Schema 轉換為 ClassProperty
        /// </summary>
        private ClassProperty? ConvertSchemaToClassProperty(string propName, Models.SwaggerV2.Schema propSchema, List<string> requiredProperties, Dictionary<string, Models.SwaggerV2.Schema> allDefinitions)
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
                classProp.Type = refTypeName;
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
                        var itemType = GetSchemaType(propSchema.Items, allDefinitions);
                        classProp.Type = $"List<{itemType}>";
                        classProp.IsArray = true;
                    }
                    else
                    {
                        classProp.Type = "List<object>";
                        classProp.IsArray = true;
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
        /// 將 OpenAPI 3.0 Schema 轉換為 ClassProperty
        /// </summary>
        private ClassProperty? ConvertOpenApiSchemaToClassProperty(string propName, Models.OpenApiV3.Schema propSchema, List<string> requiredProperties, Dictionary<string, Models.OpenApiV3.Schema> allSchemas)
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
                classProp.Type = refTypeName;
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
                        var itemType = GetOpenApiSchemaType(propSchema.Items, allSchemas);
                        classProp.Type = $"List<{itemType}>";
                        classProp.IsArray = true;
                    }
                    else
                    {
                        classProp.Type = "List<object>";
                        classProp.IsArray = true;
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
        /// 從 $ref 中提取類型名稱
        /// </summary>
        private string ExtractTypeNameFromRef(string reference)
        {
            if (string.IsNullOrEmpty(reference)) return "object";

            // 處理 Swagger 2.0: "#/definitions/User"
            // 處理 OpenAPI 3.0: "#/components/schemas/User"
            var parts = reference.Split('/');
            return parts.LastOrDefault() ?? "object";
        }

        /// <summary>
        /// 獲取 Swagger 2.0 Schema 的 C# 類型
        /// </summary>
        private string GetSchemaType(Models.SwaggerV2.Schema schema, Dictionary<string, Models.SwaggerV2.Schema> allDefinitions)
        {
            if (schema == null) return "object";

            if (!string.IsNullOrEmpty(schema.Ref))
            {
                return ExtractTypeNameFromRef(schema.Ref);
            }

            var schemaType = schema.Type?.ToLower();
            return schemaType switch
            {
                "string" => "string",
                "integer" => schema.Format == "int64" ? "long" : "int",
                "number" => schema.Format == "float" ? "float" : "decimal",
                "boolean" => "bool",
                "array" => schema.Items != null ? $"List<{GetSchemaType(schema.Items, allDefinitions)}>" : "List<object>",
                "object" => "object",
                _ => "object"
            };
        }

        /// <summary>
        /// 獲取 OpenAPI 3.0 Schema 的 C# 類型
        /// </summary>
        private string GetOpenApiSchemaType(Models.OpenApiV3.Schema schema, Dictionary<string, Models.OpenApiV3.Schema> allSchemas)
        {
            if (schema == null) return "object";

            if (!string.IsNullOrEmpty(schema.Ref))
            {
                return ExtractTypeNameFromRef(schema.Ref);
            }

            var schemaType = schema.Type?.ToLower();
            return schemaType switch
            {
                "string" => "string",
                "integer" => schema.Format == "int64" ? "long" : "int",
                "number" => schema.Format == "float" ? "float" : "decimal",
                "boolean" => "bool",
                "array" => schema.Items != null ? $"List<{GetOpenApiSchemaType(schema.Items, allSchemas)}>" : "List<object>",
                "object" => "object",
                _ => "object"
            };
        }

        /// <summary>
        /// 將 Swagger 2.0 PathItem 轉換為 SwaggerEndpoint 列表
        /// </summary>
        private List<SwaggerEndpoint> ConvertPathItemToEndpoints(string path, Models.SwaggerV2.PathItem pathItem, Models.SwaggerV2.SwaggerDocument swaggerDoc, SwaggerApiInfo apiInfo)
        {
            var endpoints = new List<SwaggerEndpoint>();

            var operations = new Dictionary<string, Models.SwaggerV2.Operation?>
            {
                { "GET", pathItem.Get },
                { "POST", pathItem.Post },
                { "PUT", pathItem.Put },
                { "DELETE", pathItem.Delete },
                { "OPTIONS", pathItem.Options },
                { "HEAD", pathItem.Head },
                { "PATCH", pathItem.Patch }
            };

            foreach (var operation in operations)
            {
                if (operation.Value != null)
                {
                    var endpoint = ConvertOperationToEndpoint(path, operation.Key, operation.Value, swaggerDoc, apiInfo);
                    if (endpoint != null)
                    {
                        endpoints.Add(endpoint);
                    }
                }
            }

            return endpoints;
        }

        /// <summary>
        /// 將 OpenAPI 3.0 PathItem 轉換為 SwaggerEndpoint 列表
        /// </summary>
        private List<SwaggerEndpoint> ConvertOpenApiPathItemToEndpoints(string path, Models.OpenApiV3.PathItem pathItem, Models.OpenApiV3.OpenApiDocument openApiDoc, SwaggerApiInfo apiInfo)
        {
            var endpoints = new List<SwaggerEndpoint>();

            var operations = new Dictionary<string, Models.OpenApiV3.Operation?>
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
                    var endpoint = ConvertOpenApiOperationToEndpoint(path, operation.Key, operation.Value, openApiDoc, apiInfo);
                    if (endpoint != null)
                    {
                        endpoints.Add(endpoint);
                    }
                }
            }

            return endpoints;
        }

        /// <summary>
        /// 將 Swagger 2.0 Operation 轉換為 SwaggerEndpoint
        /// </summary>
        private SwaggerEndpoint? ConvertOperationToEndpoint(string path, string httpMethod, Models.SwaggerV2.Operation operation, Models.SwaggerV2.SwaggerDocument swaggerDoc, SwaggerApiInfo apiInfo)
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
                    var endpointParam = ConvertParameterToEndpointParameter(param, swaggerDoc, endpoint.OperationId, apiInfo);
                    if (endpointParam != null)
                    {
                        endpoint.Parameters.Add(endpointParam);
                    }
                }
            }

            // 轉換回應
            endpoint.ResponseType = ConvertResponsesToResponseType(operation.Responses, swaggerDoc, endpoint.OperationId, apiInfo);

            return endpoint;
        }

        /// <summary>
        /// 將 OpenAPI 3.0 Operation 轉換為 SwaggerEndpoint
        /// </summary>
        private SwaggerEndpoint? ConvertOpenApiOperationToEndpoint(string path, string httpMethod, Models.OpenApiV3.Operation operation, Models.OpenApiV3.OpenApiDocument openApiDoc, SwaggerApiInfo apiInfo)
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
                    var endpointParam = ConvertOpenApiParameterToEndpointParameter(param, openApiDoc, endpoint.OperationId, apiInfo);
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
            endpoint.ResponseType = ConvertOpenApiResponsesToResponseType(operation.Responses, openApiDoc, endpoint.OperationId, apiInfo);

            return endpoint;
        }

        // TODO: 實作剩餘的轉換方法
        // - ConvertParameterToEndpointParameter
        // - ConvertOpenApiParameterToEndpointParameter  
        // - ConvertRequestBodyToEndpointParameter
        // - ConvertResponsesToResponseType
        // - ConvertOpenApiResponsesToResponseType

        private EndpointParameter? ConvertParameterToEndpointParameter(Models.SwaggerV2.Parameter param, Models.SwaggerV2.SwaggerDocument swaggerDoc, string operationId, SwaggerApiInfo apiInfo)
        {
            // 簡化實作，詳細邏輯可以後續完善
            return new EndpointParameter
            {
                Name = param.Name,
                Location = param.In,
                Type = param.Type ?? "object",
                IsRequired = param.Required,
                Description = param.Description ?? ""
            };
        }

        private EndpointParameter? ConvertOpenApiParameterToEndpointParameter(Models.OpenApiV3.Parameter param, Models.OpenApiV3.OpenApiDocument openApiDoc, string operationId, SwaggerApiInfo apiInfo)
        {
            // 簡化實作，詳細邏輯可以後續完善
            return new EndpointParameter
            {
                Name = param.Name,
                Location = param.In,
                Type = GetOpenApiSchemaType(param.Schema, openApiDoc.Components?.Schemas ?? new Dictionary<string, Models.OpenApiV3.Schema>()),
                IsRequired = param.Required,
                Description = param.Description ?? ""
            };
        }

        private EndpointParameter? ConvertRequestBodyToEndpointParameter(Models.OpenApiV3.RequestBody requestBody, Models.OpenApiV3.OpenApiDocument openApiDoc, string operationId, SwaggerApiInfo apiInfo)
        {
            // 簡化實作，詳細邏輯可以後續完善
            var jsonContent = requestBody.Content.FirstOrDefault(c => c.Key.Contains("json"));
            if (jsonContent.Value?.Schema != null)
            {
                return new EndpointParameter
                {
                    Name = "body",
                    Location = "body",
                    Type = GetOpenApiSchemaType(jsonContent.Value.Schema, openApiDoc.Components?.Schemas ?? new Dictionary<string, Models.OpenApiV3.Schema>()),
                    IsRequired = requestBody.Required,
                    Description = requestBody.Description ?? ""
                };
            }
            return null;
        }

        private ResponseType ConvertResponsesToResponseType(Dictionary<string, Models.SwaggerV2.Response> responses, Models.SwaggerV2.SwaggerDocument swaggerDoc, string operationId, SwaggerApiInfo apiInfo)
        {
            // 簡化實作，詳細邏輯可以後續完善
            var responseType = new ResponseType();
            var successResponse = responses.FirstOrDefault(r => r.Key.StartsWith("2"));
            if (successResponse.Value?.Schema != null)
            {
                responseType.Type = GetSchemaType(successResponse.Value.Schema, swaggerDoc.Definitions ?? new Dictionary<string, Models.SwaggerV2.Schema>());
            }
            return responseType;
        }

        private ResponseType ConvertOpenApiResponsesToResponseType(Dictionary<string, Models.OpenApiV3.Response> responses, Models.OpenApiV3.OpenApiDocument openApiDoc, string operationId, SwaggerApiInfo apiInfo)
        {
            // 簡化實作，詳細邏輯可以後續完善
            var responseType = new ResponseType();
            var successResponse = responses.FirstOrDefault(r => r.Key.StartsWith("2"));
            if (successResponse.Value?.Content != null)
            {
                var jsonContent = successResponse.Value.Content.FirstOrDefault(c => c.Key.Contains("json"));
                if (jsonContent.Value?.Schema != null)
                {
                    responseType.Type = GetOpenApiSchemaType(jsonContent.Value.Schema, openApiDoc.Components?.Schemas ?? new Dictionary<string, Models.OpenApiV3.Schema>());
                }
            }
            return responseType;
        }
    }
}
