using T1.SwaggerEx.Models;
using T1.SwaggerEx.Models.SwaggerV2;
using System;
using System.Collections.Generic;
using System.Linq;

namespace T1.SwaggerEx.Services
{
    /// <summary>
    /// 專門負責將 Swagger 2.0 模型轉換為 SwaggerApiInfo
    /// </summary>
    public class SwaggerV2ModelConverter
    {
        /// <summary>
        /// 將 Swagger 2.0 文檔轉換為 SwaggerApiInfo
        /// </summary>
        /// <param name="swaggerDoc">Swagger 2.0 文檔</param>
        /// <returns>轉換後的 API 資訊</returns>
        public SwaggerApiInfo Convert(SwaggerDocument swaggerDoc)
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
            BuildBaseUrl(swaggerDoc, apiInfo);

            // 轉換 definitions 為 ClassDefinitions
            ConvertDefinitions(swaggerDoc, apiInfo);

            // 轉換 paths 為 Endpoints
            ConvertPaths(swaggerDoc, apiInfo);

            return apiInfo;
        }

        /// <summary>
        /// 建構 Base URL
        /// </summary>
        private void BuildBaseUrl(SwaggerDocument swaggerDoc, SwaggerApiInfo apiInfo)
        {
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
        }

        /// <summary>
        /// 轉換 definitions 為 ClassDefinitions
        /// </summary>
        private void ConvertDefinitions(SwaggerDocument swaggerDoc, SwaggerApiInfo apiInfo)
        {
            if (swaggerDoc.Definitions == null) return;

            foreach (var definition in swaggerDoc.Definitions)
            {
                var classDefinition = ConvertSchemaToClassDefinition(definition.Key, definition.Value, swaggerDoc.Definitions);
                if (classDefinition != null)
                {
                    apiInfo.ClassDefinitions[definition.Key] = classDefinition;
                }
            }
        }

        /// <summary>
        /// 轉換 paths 為 Endpoints
        /// </summary>
        private void ConvertPaths(SwaggerDocument swaggerDoc, SwaggerApiInfo apiInfo)
        {
            if (swaggerDoc.Paths == null) return;

            foreach (var path in swaggerDoc.Paths)
            {
                var pathEndpoints = ConvertPathItemToEndpoints(path.Key, path.Value, swaggerDoc, apiInfo);
                apiInfo.Endpoints.AddRange(pathEndpoints);
            }
        }

        /// <summary>
        /// 將 Schema 轉換為 ClassDefinition
        /// </summary>
        private ClassDefinition? ConvertSchemaToClassDefinition(string className, Schema schema, Dictionary<string, Schema> allDefinitions)
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
                classDefinition.IsNumericEnum = IsNumericEnum(schema.Enum);

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
        /// 將 Schema 轉換為 ClassProperty
        /// </summary>
        private ClassProperty? ConvertSchemaToClassProperty(string propName, Schema propSchema, List<string> requiredProperties, Dictionary<string, Schema> allDefinitions)
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
                if (allDefinitions.TryGetValue(refTypeName, out var refSchema) && 
                    refSchema.Enum != null && refSchema.Enum.Count > 0 &&
                    IsNumericEnum(refSchema.Enum))
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
                        var itemType = GetSchemaType(propSchema.Items, allDefinitions);
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
        private List<SwaggerEndpoint> ConvertPathItemToEndpoints(string path, PathItem pathItem, SwaggerDocument swaggerDoc, SwaggerApiInfo apiInfo)
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
        /// 將 Operation 轉換為 SwaggerEndpoint
        /// </summary>
        private SwaggerEndpoint? ConvertOperationToEndpoint(string path, string httpMethod, Operation operation, SwaggerDocument swaggerDoc, SwaggerApiInfo apiInfo)
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
        /// 將 Parameter 轉換為 EndpointParameter
        /// </summary>
        private EndpointParameter? ConvertParameterToEndpointParameter(Parameter param, SwaggerDocument swaggerDoc, string operationId, SwaggerApiInfo apiInfo)
        {
            if (param == null) return null;

            var endpointParam = new EndpointParameter
            {
                Name = param.Name,
                Location = param.In,
                IsRequired = param.Required,
                Description = param.Description ?? ""
            };

            // 處理 body 參數
            if (param.In.ToLower() == "body" && param.Schema != null)
            {
                endpointParam.Type = GetSchemaType(param.Schema, swaggerDoc.Definitions ?? new Dictionary<string, Schema>());

                // 特別處理陣列類型的 body 參數（如 createUsersWithListInput）
                if (param.Schema.Type == "array" && param.Schema.Items != null)
                {
                    var itemType = GetSchemaType(param.Schema.Items, swaggerDoc.Definitions ?? new Dictionary<string, Schema>());
                    endpointParam.Type = $"List<{itemType}>";
                }
            }
            else
            {
                // 處理非 body 參數
                endpointParam.Type = param.Type ?? "object";

                // 處理陣列類型的查詢參數
                if (param.Type == "array" && param.Items != null)
                {
                    var itemType = GetItemType(param.Items);
                    endpointParam.Type = $"List<{itemType}>";
                }
            }

            return endpointParam;
        }

        /// <summary>
        /// 將 Responses 轉換為 ResponseType
        /// </summary>
        private ResponseType ConvertResponsesToResponseType(Dictionary<string, Response> responses, SwaggerDocument swaggerDoc, string operationId, SwaggerApiInfo apiInfo)
        {
            var responseType = new ResponseType();

            // 尋找成功回應（2xx）
            var successResponse = responses.FirstOrDefault(r => r.Key.StartsWith("2"));
            if (successResponse.Value != null)
            {
                responseType.Description = successResponse.Value.Description;

                if (successResponse.Value.Schema != null)
                {
                    var schema = successResponse.Value.Schema;

                    // 處理陣列回應
                    if (schema.Type == "array" && schema.Items != null)
                    {
                        responseType.IsArray = true;
                        responseType.Type = GetSchemaType(schema.Items, swaggerDoc.Definitions ?? new Dictionary<string, Schema>());
                    }
                    else
                    {
                        responseType.Type = GetSchemaType(schema, swaggerDoc.Definitions ?? new Dictionary<string, Schema>());
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
        private string GetSchemaType(Schema schema, Dictionary<string, Schema> allDefinitions)
        {
            if (schema == null) return "object";

            if (!string.IsNullOrEmpty(schema.Ref))
            {
                var refTypeName = ExtractTypeNameFromRef(schema.Ref);
                
                // Check if the referenced type is a numeric enum, if so use int instead
                if (allDefinitions.TryGetValue(refTypeName, out var refSchema) && 
                    refSchema.Enum != null && refSchema.Enum.Count > 0 &&
                    IsNumericEnum(refSchema.Enum))
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
                "array" => schema.Items != null ? $"List<{GetSchemaType(schema.Items, allDefinitions)}>" : "List<object>",
                "object" => "object",
                _ => "object"
            };
        }

        /// <summary>
        /// 獲取 Items 的 C# 類型
        /// </summary>
        private string GetItemType(Items items)
        {
            if (items == null) return "object";

            return items.Type?.ToLower() switch
            {
                "string" => "string",
                "integer" => items.Format == "int64" ? "long" : "int",
                "number" => items.Format == "float" ? "float" : "decimal",
                "boolean" => "bool",
                _ => "object"
            };
        }

        /// <summary>
        /// 檢查 enum 值是否都是數字
        /// </summary>
        private bool IsNumericEnum(List<object> enumValues)
        {
            if (enumValues == null || enumValues.Count == 0)
                return false;

            return enumValues.All(e =>
            {
                if (e == null) return false;

                // Handle JsonElement (when deserializing from JSON)
                if (e is System.Text.Json.JsonElement jsonElement)
                {
                    return jsonElement.ValueKind == System.Text.Json.JsonValueKind.Number;
                }

                // Handle direct numeric types
                return e is int || e is long || e is decimal || e is double || e is float ||
                       e is short || e is byte || e is sbyte || e is uint || e is ulong || e is ushort;
            });
        }

        /// <summary>
        /// 從 $ref 中提取類型名稱
        /// </summary>
        private string ExtractTypeNameFromRef(string reference)
        {
            if (string.IsNullOrEmpty(reference)) return "object";

            // 處理 Swagger 2.0: "#/definitions/User"
            var parts = reference.Split('/');
            return parts.LastOrDefault() ?? "object";
        }
    }
}
