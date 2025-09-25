using HtmlAgilityPack;
using CodeBoyLib.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace CodeBoyLib.Services
{
    /// <summary>
    /// 新版 SwaggerUiParser，使用強型別反序列化取代動態 JSON 解析
    /// </summary>
    public class SwaggerUiParser
    {
        private readonly HttpClient _httpClient;
        private readonly SwaggerDocumentDeserializer _deserializer;
        private readonly SwaggerV2ModelConverter _v2Converter;
        private readonly SwaggerV3ModelConverter _v3Converter;

        public SwaggerUiParser(HttpClient? httpClient = null)
        {
            _httpClient = httpClient ?? new HttpClient();
            _deserializer = new SwaggerDocumentDeserializer();
            _v2Converter = new SwaggerV2ModelConverter();
            _v3Converter = new SwaggerV3ModelConverter();
        }

        /// <summary>
        /// 從 URL 獲取並解析 Swagger JSON
        /// </summary>
        /// <param name="swaggerUrl">Swagger JSON URL</param>
        /// <returns>解析後的 API 資訊</returns>
        public async Task<SwaggerApiInfo> ParseFromUrlAsync(string swaggerUrl)
        {
            if (string.IsNullOrWhiteSpace(swaggerUrl))
                throw new ArgumentException("Swagger URL cannot be null or empty", nameof(swaggerUrl));

            try
            {
                Console.WriteLine($"🔍 Fetching Swagger JSON from: {swaggerUrl}");
                
                var jsonContent = await _httpClient.GetStringAsync(swaggerUrl);
                
                Console.WriteLine($"✅ Successfully fetched JSON content ({jsonContent.Length:N0} characters)");
                
                return ParseFromJson(jsonContent);
            }
            catch (HttpRequestException ex)
            {
                throw new InvalidOperationException($"Failed to fetch Swagger JSON from {swaggerUrl}", ex);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Error parsing Swagger from URL {swaggerUrl}", ex);
            }
        }

        /// <summary>
        /// 從 JSON 字符串解析 Swagger 文檔
        /// </summary>
        /// <param name="jsonContent">JSON 內容</param>
        /// <returns>解析後的 API 資訊</returns>
        public SwaggerApiInfo ParseFromJson(string jsonContent)
        {
            if (string.IsNullOrWhiteSpace(jsonContent))
                throw new ArgumentException("JSON content cannot be null or empty", nameof(jsonContent));

            try
            {
                Console.WriteLine("🔄 Attempting to deserialize Swagger/OpenAPI document...");
                
                var deserializationResult = _deserializer.Deserialize(jsonContent);
                
                if (!deserializationResult.IsSuccess)
                {
                    throw new InvalidOperationException($"Failed to deserialize Swagger/OpenAPI document: {deserializationResult.ErrorMessage}");
                }

                Console.WriteLine($"✅ Successfully deserialized as {deserializationResult.Version}");

                SwaggerApiInfo apiInfo;

                if (deserializationResult.Version == SwaggerDocumentDeserializer.SwaggerVersion.Swagger2 && 
                    deserializationResult.SwaggerV2Document != null)
                {
                    Console.WriteLine("🔄 Converting Swagger 2.0 document to ApiInfo using V2 converter...");
                    apiInfo = _v2Converter.Convert(deserializationResult.SwaggerV2Document);
                }
                else if (deserializationResult.Version == SwaggerDocumentDeserializer.SwaggerVersion.OpenApi3 && 
                         deserializationResult.OpenApiV3Document != null)
                {
                    Console.WriteLine("🔄 Converting OpenAPI 3.0 document to ApiInfo using V3 converter...");
                    apiInfo = _v3Converter.Convert(deserializationResult.OpenApiV3Document);
                }
                else
                {
                    throw new InvalidOperationException("Unexpected deserialization result state");
                }

                Console.WriteLine($"✅ Conversion completed:");
                Console.WriteLine($"   📊 API: {apiInfo.Title} v{apiInfo.Version}");
                Console.WriteLine($"   🎯 Endpoints: {apiInfo.Endpoints.Count}");
                Console.WriteLine($"   📚 Classes: {apiInfo.ClassDefinitions.Count}");

                return apiInfo;
            }
            catch (Exception ex) when (!(ex is InvalidOperationException))
            {
                throw new InvalidOperationException($"Error parsing JSON content: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// 從 Swagger UI 頁面自動發現並解析 JSON
        /// </summary>
        /// <param name="swaggerUiUrl">Swagger UI 頁面 URL</param>
        /// <returns>解析後的 API 資訊</returns>
        public async Task<SwaggerApiInfo> ParseFromSwaggerUiAsync(string swaggerUiUrl)
        {
            if (string.IsNullOrWhiteSpace(swaggerUiUrl))
                throw new ArgumentException("Swagger UI URL cannot be null or empty", nameof(swaggerUiUrl));

            try
            {
                Console.WriteLine($"🔍 Fetching Swagger UI page: {swaggerUiUrl}");
                
                var htmlContent = await _httpClient.GetStringAsync(swaggerUiUrl);
                
                Console.WriteLine("🔄 Searching for Swagger JSON URL in HTML...");
                
                var swaggerJsonUrl = ExtractSwaggerJsonUrl(htmlContent, swaggerUiUrl);
                
                if (string.IsNullOrEmpty(swaggerJsonUrl))
                {
                    throw new InvalidOperationException("Could not find Swagger JSON URL in the provided Swagger UI page");
                }

                Console.WriteLine($"✅ Found Swagger JSON URL: {swaggerJsonUrl}");
                
                return await ParseFromUrlAsync(swaggerJsonUrl);
            }
            catch (HttpRequestException ex)
            {
                throw new InvalidOperationException($"Failed to fetch Swagger UI page from {swaggerUiUrl}", ex);
            }
            catch (Exception ex) when (!(ex is InvalidOperationException))
            {
                throw new InvalidOperationException($"Error parsing Swagger UI from {swaggerUiUrl}", ex);
            }
        }

        /// <summary>
        /// 從 HTML 內容中提取 Swagger JSON URL
        /// </summary>
        /// <param name="htmlContent">HTML 內容</param>
        /// <param name="baseUrl">基礎 URL</param>
        /// <returns>Swagger JSON URL</returns>
        private string? ExtractSwaggerJsonUrl(string htmlContent, string baseUrl)
        {
            try
            {
                var doc = new HtmlDocument();
                doc.LoadHtml(htmlContent);

                // 方法 1: 查找 script 標籤中的 url 參數
                var scriptNodes = doc.DocumentNode.SelectNodes("//script");
                if (scriptNodes != null)
                {
                    foreach (var script in scriptNodes)
                    {
                        var scriptText = script.InnerText;
                        if (string.IsNullOrEmpty(scriptText)) continue;

                        // 查找 SwaggerUIBundle 或類似的配置
                        var urlMatches = Regex.Matches(scriptText, @"[""']?url[""']?\s*:\s*[""']([^""']+)[""']", RegexOptions.IgnoreCase);
                        foreach (Match match in urlMatches)
                        {
                            var url = match.Groups[1].Value;
                            if (IsSwaggerJsonUrl(url))
                            {
                                return ResolveUrl(url, baseUrl);
                            }
                        }

                        // 查找直接的 JSON URL 引用
                        var directUrlMatches = Regex.Matches(scriptText, @"[""']([^""']*\.json)[""']", RegexOptions.IgnoreCase);
                        foreach (Match match in directUrlMatches)
                        {
                            var url = match.Groups[1].Value;
                            if (IsSwaggerJsonUrl(url))
                            {
                                return ResolveUrl(url, baseUrl);
                            }
                        }
                    }
                }

                // 方法 2: 查找 data-* 屬性
                var elementsWithData = doc.DocumentNode.SelectNodes("//*[@data-swagger-url or @data-url]");
                if (elementsWithData != null)
                {
                    foreach (var element in elementsWithData)
                    {
                        var dataUrl = element.GetAttributeValue("data-swagger-url", "") ?? 
                                     element.GetAttributeValue("data-url", "");
                        if (!string.IsNullOrEmpty(dataUrl) && IsSwaggerJsonUrl(dataUrl))
                        {
                            return ResolveUrl(dataUrl, baseUrl);
                        }
                    }
                }

                // 方法 3: 嘗試常見的相對路徑
                var commonPaths = new[] { "swagger.json", "v2/api-docs", "v3/api-docs", "api-docs" };
                foreach (var path in commonPaths)
                {
                    var testUrl = ResolveUrl(path, baseUrl);
                    if (!string.IsNullOrEmpty(testUrl))
                    {
                        return testUrl;
                    }
                }

                return null;
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// 檢查 URL 是否可能是 Swagger JSON URL
        /// </summary>
        /// <param name="url">URL</param>
        /// <returns>是否為 Swagger JSON URL</returns>
        private bool IsSwaggerJsonUrl(string url)
        {
            if (string.IsNullOrEmpty(url)) return false;

            url = url.ToLower();
            return url.Contains("swagger") || 
                   url.Contains("api-docs") || 
                   url.Contains("openapi") ||
                   url.EndsWith(".json");
        }

        /// <summary>
        /// 解析相對 URL 為絕對 URL
        /// </summary>
        /// <param name="url">URL</param>
        /// <param name="baseUrl">基礎 URL</param>
        /// <returns>絕對 URL</returns>
        private string ResolveUrl(string url, string baseUrl)
        {
            if (string.IsNullOrEmpty(url)) return "";

            try
            {
                if (Uri.IsWellFormedUriString(url, UriKind.Absolute))
                {
                    return url;
                }

                var baseUri = new Uri(baseUrl);
                var resolvedUri = new Uri(baseUri, url);
                return resolvedUri.ToString();
            }
            catch
            {
                return "";
            }
        }

        public void Dispose()
        {
            _httpClient?.Dispose();
        }
    }
}
