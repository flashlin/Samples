using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Http;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.ComponentModel.DataAnnotations;

namespace SteropesSDK
{
    /// <summary>
    /// Test class for simplified nullable return logic
    /// </summary>
    public class TestResponse
    {
        [JsonPropertyName("id")]
        public int Id { get; set; }

        [JsonPropertyName("name")]
        public string? Name { get; set; }
    }

    /// <summary>
    /// Steropes API Client
    /// </summary>
    public class SteropesClient : IDisposable
    {
        private readonly HttpClient _httpClient;
        private readonly string _baseUrl;

        public SteropesClient(HttpClient httpClient, string baseUrl)
        {
            _httpClient = httpClient ?? throw new ArgumentNullException(nameof(httpClient));
            _baseUrl = baseUrl.TrimEnd('/');
        }

        /// <summary>
        /// Test method with simplified nullable return
        /// </summary>
        /// <returns>Task<TestResponse?></returns>
        public async Task<TestResponse?> GetTestResponse()
        {
            var url = $"/test";
            var request = new HttpRequestMessage(HttpMethod.Get, _baseUrl + url);

            var response = await _httpClient.SendAsync(request);
            response.EnsureSuccessStatusCode();

            var responseContent = await response.Content.ReadAsStringAsync();

            if (string.IsNullOrEmpty(responseContent))
                return null;

            return JsonSerializer.Deserialize<TestResponse?>(responseContent);
        }

        /// <summary>
        /// Test method with list return type
        /// </summary>
        /// <returns>Task<List<TestResponse>?></returns>
        public async Task<List<TestResponse>?> GetTestList()
        {
            var url = $"/test-list";
            var request = new HttpRequestMessage(HttpMethod.Get, _baseUrl + url);

            var response = await _httpClient.SendAsync(request);
            response.EnsureSuccessStatusCode();

            var responseContent = await response.Content.ReadAsStringAsync();

            if (string.IsNullOrEmpty(responseContent))
                return null;

            return JsonSerializer.Deserialize<List<TestResponse>?>(responseContent);
        }

        /// <summary>
        /// Dispose resources
        /// </summary>
        public void Dispose()
        {
            _httpClient?.Dispose();
        }
    }
}
