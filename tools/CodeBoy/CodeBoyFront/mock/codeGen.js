// Mock data for code generation endpoints
export default [
  {
    url: '/api/codegen/genWebApiClient',
    method: 'post',
    response: ({ body }) => {
      const { swaggerUrl, sdkName } = body;
      
      // Validate required fields
      if (!swaggerUrl || !sdkName) {
        return 'Error: SwaggerUrl and SdkName are required';
      }

      // Simulate code generation
      const generatedCode = `// Generated SDK: ${sdkName}
// Source: ${swaggerUrl}
// Generated at: ${new Date().toISOString()}

using System;
using System.Net.Http;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace ${sdkName}
{
    /// <summary>
    /// Auto-generated API client for ${sdkName}
    /// </summary>
    public class ${sdkName}Client
    {
        private readonly HttpClient _httpClient;
        private readonly string _baseUrl;

        public ${sdkName}Client(HttpClient httpClient, string baseUrl)
        {
            _httpClient = httpClient ?? throw new ArgumentNullException(nameof(httpClient));
            _baseUrl = baseUrl ?? throw new ArgumentNullException(nameof(baseUrl));
        }

        // TODO: Add API methods based on Swagger definition
        // Source: ${swaggerUrl}
        
        public async Task<string> GetHealthAsync()
        {
            var response = await _httpClient.GetAsync($"{_baseUrl}/health");
            response.EnsureSuccessStatusCode();
            return await response.Content.ReadAsStringAsync();
        }
    }
}`;

      return generatedCode;
    },
  },
];
