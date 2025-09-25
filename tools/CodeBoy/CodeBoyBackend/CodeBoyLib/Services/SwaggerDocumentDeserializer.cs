using CodeBoyLib.Models;
using CodeBoyLib.Models.SwaggerV2;
using CodeBoyLib.Models.OpenApiV3;
using System;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace CodeBoyLib.Services
{
    /// <summary>
    /// 智能反序列化器，自動偵測並解析 Swagger 2.0 或 OpenAPI 3.0 文檔
    /// </summary>
    public class SwaggerDocumentDeserializer
    {
        public enum SwaggerVersion
        {
            Unknown,
            Swagger2,
            OpenApi3
        }

        public class DeserializationResult
        {
            public SwaggerVersion Version { get; set; } = SwaggerVersion.Unknown;
            public SwaggerDocument? SwaggerV2Document { get; set; }
            public OpenApiDocument? OpenApiV3Document { get; set; }
            public string? ErrorMessage { get; set; }
            public bool IsSuccess => Version != SwaggerVersion.Unknown && ErrorMessage == null;
        }

        /// <summary>
        /// 嘗試反序列化 JSON 為 Swagger/OpenAPI 文檔
        /// 先嘗試 Swagger 2.0，失敗後嘗試 OpenAPI 3.0
        /// </summary>
        /// <param name="jsonContent">JSON 內容</param>
        /// <returns>反序列化結果</returns>
        public DeserializationResult Deserialize(string jsonContent)
        {
            if (string.IsNullOrWhiteSpace(jsonContent))
            {
                return new DeserializationResult
                {
                    ErrorMessage = "JSON content is null or empty"
                };
            }

            // 首先嘗試快速偵測版本
            var detectedVersion = DetectSwaggerVersion(jsonContent);
            
            // 根據偵測結果選擇反序列化順序
            if (detectedVersion == SwaggerVersion.Swagger2)
            {
                var v2Result = TryDeserializeSwaggerV2(jsonContent);
                if (v2Result.IsSuccess) return v2Result;

                // 如果 v2 失敗，嘗試 v3
                var v3Result = TryDeserializeOpenApiV3(jsonContent);
                if (v3Result.IsSuccess) return v3Result;

                return new DeserializationResult
                {
                    ErrorMessage = $"Failed to parse as Swagger 2.0: {v2Result.ErrorMessage}. Also failed as OpenAPI 3.0: {v3Result.ErrorMessage}"
                };
            }
            else if (detectedVersion == SwaggerVersion.OpenApi3)
            {
                var v3Result = TryDeserializeOpenApiV3(jsonContent);
                if (v3Result.IsSuccess) return v3Result;

                // 如果 v3 失敗，嘗試 v2
                var v2Result = TryDeserializeSwaggerV2(jsonContent);
                if (v2Result.IsSuccess) return v2Result;

                return new DeserializationResult
                {
                    ErrorMessage = $"Failed to parse as OpenAPI 3.0: {v3Result.ErrorMessage}. Also failed as Swagger 2.0: {v2Result.ErrorMessage}"
                };
            }
            else
            {
                // 版本未知，按預設順序嘗試：先 v2.0 再 v3.0
                var v2Result = TryDeserializeSwaggerV2(jsonContent);
                if (v2Result.IsSuccess) return v2Result;

                var v3Result = TryDeserializeOpenApiV3(jsonContent);
                if (v3Result.IsSuccess) return v3Result;

                return new DeserializationResult
                {
                    ErrorMessage = $"Failed to parse as both Swagger 2.0 and OpenAPI 3.0. Swagger 2.0 error: {v2Result.ErrorMessage}. OpenAPI 3.0 error: {v3Result.ErrorMessage}"
                };
            }
        }

        /// <summary>
        /// 快速偵測 Swagger 版本（不進行完整反序列化）
        /// </summary>
        /// <param name="jsonContent">JSON 內容</param>
        /// <returns>偵測到的版本</returns>
        private SwaggerVersion DetectSwaggerVersion(string jsonContent)
        {
            try
            {
                // 使用簡單字符串檢查來快速偵測版本
                if (jsonContent.Contains("\"swagger\"") && jsonContent.Contains("\"2.0\""))
                {
                    return SwaggerVersion.Swagger2;
                }

                if (jsonContent.Contains("\"openapi\"") && (jsonContent.Contains("\"3.0") || jsonContent.Contains("\"3.1")))
                {
                    return SwaggerVersion.OpenApi3;
                }

                // 更詳細的檢查（避免完整 JSON 解析的效能成本）
                if (jsonContent.Contains("\"swagger\":") || jsonContent.Contains("\"definitions\":"))
                {
                    return SwaggerVersion.Swagger2;
                }

                if (jsonContent.Contains("\"openapi\":") || jsonContent.Contains("\"components\":"))
                {
                    return SwaggerVersion.OpenApi3;
                }

                return SwaggerVersion.Unknown;
            }
            catch
            {
                return SwaggerVersion.Unknown;
            }
        }

        /// <summary>
        /// 嘗試反序列化為 Swagger 2.0 文檔
        /// </summary>
        /// <param name="jsonContent">JSON 內容</param>
        /// <returns>反序列化結果</returns>
        private DeserializationResult TryDeserializeSwaggerV2(string jsonContent)
        {
            try
            {
                var options = new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true,
                    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
                };

                var document = JsonSerializer.Deserialize<SwaggerDocument>(jsonContent, options);
                
                if (document == null)
                {
                    return new DeserializationResult
                    {
                        ErrorMessage = "Deserialization resulted in null SwaggerDocument"
                    };
                }

                // 驗證這確實是 Swagger 2.0 文檔
                if (string.IsNullOrEmpty(document.Swagger))
                {
                    return new DeserializationResult
                    {
                        ErrorMessage = "Missing 'swagger' field, not a valid Swagger 2.0 document"
                    };
                }

                if (!document.Swagger.StartsWith("2."))
                {
                    return new DeserializationResult
                    {
                        ErrorMessage = $"Swagger version '{document.Swagger}' is not 2.x"
                    };
                }

                return new DeserializationResult
                {
                    Version = SwaggerVersion.Swagger2,
                    SwaggerV2Document = document
                };
            }
            catch (JsonException ex)
            {
                return new DeserializationResult
                {
                    ErrorMessage = $"JSON parsing error for Swagger 2.0: {ex.Message}"
                };
            }
            catch (Exception ex)
            {
                return new DeserializationResult
                {
                    ErrorMessage = $"Unexpected error parsing Swagger 2.0: {ex.Message}"
                };
            }
        }

        /// <summary>
        /// 嘗試反序列化為 OpenAPI 3.0 文檔
        /// </summary>
        /// <param name="jsonContent">JSON 內容</param>
        /// <returns>反序列化結果</returns>
        private DeserializationResult TryDeserializeOpenApiV3(string jsonContent)
        {
            try
            {
                var options = new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true,
                    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
                };

                var document = JsonSerializer.Deserialize<OpenApiDocument>(jsonContent, options);
                
                if (document == null)
                {
                    return new DeserializationResult
                    {
                        ErrorMessage = "Deserialization resulted in null OpenApiDocument"
                    };
                }

                // 驗證這確實是 OpenAPI 3.x 文檔
                if (string.IsNullOrEmpty(document.OpenApi))
                {
                    return new DeserializationResult
                    {
                        ErrorMessage = "Missing 'openapi' field, not a valid OpenAPI 3.0 document"
                    };
                }

                if (!document.OpenApi.StartsWith("3."))
                {
                    return new DeserializationResult
                    {
                        ErrorMessage = $"OpenAPI version '{document.OpenApi}' is not 3.x"
                    };
                }

                return new DeserializationResult
                {
                    Version = SwaggerVersion.OpenApi3,
                    OpenApiV3Document = document
                };
            }
            catch (JsonException ex)
            {
                return new DeserializationResult
                {
                    ErrorMessage = $"JSON parsing error for OpenAPI 3.0: {ex.Message}"
                };
            }
            catch (Exception ex)
            {
                return new DeserializationResult
                {
                    ErrorMessage = $"Unexpected error parsing OpenAPI 3.0: {ex.Message}"
                };
            }
        }

        /// <summary>
        /// 將 Swagger 2.0 文檔轉換為統一的 SwaggerApiInfo
        /// </summary>
        /// <param name="swaggerDoc">Swagger 2.0 文檔</param>
        /// <returns>統一的 API 資訊</returns>
        public SwaggerApiInfo ConvertToApiInfo(SwaggerDocument swaggerDoc)
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

            // TODO: 實作 paths 和 definitions 的轉換
            // 這裡需要將 SwaggerV2 的結構轉換為現有的 SwaggerApiInfo 結構
            
            return apiInfo;
        }

        /// <summary>
        /// 將 OpenAPI 3.0 文檔轉換為統一的 SwaggerApiInfo
        /// </summary>
        /// <param name="openApiDoc">OpenAPI 3.0 文檔</param>
        /// <returns>統一的 API 資訊</returns>
        public SwaggerApiInfo ConvertToApiInfo(OpenApiDocument openApiDoc)
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

            // TODO: 實作 paths 和 components/schemas 的轉換
            // 這裡需要將 OpenApiV3 的結構轉換為現有的 SwaggerApiInfo 結構
            
            return apiInfo;
        }
    }
}
