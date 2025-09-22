using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Http;
using Newtonsoft.Json;
using System.ComponentModel.DataAnnotations;

namespace PetStoreSDK
{
    /// <summary>
    /// 
    /// </summary>
    public class ApiResponse
    {
        [JsonProperty("code")]
        public int? Code { get; set; }

        [JsonProperty("type")]
        public string Type { get; set; }

        [JsonProperty("message")]
        public string Message { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public class Category
    {
        [JsonProperty("id")]
        public long? Id { get; set; }

        [JsonProperty("name")]
        public string Name { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public class Pet
    {
        [JsonProperty("id")]
        public long? Id { get; set; }

        [JsonProperty("category")]
        public Category Category { get; set; }

        [Required]
        [JsonProperty("name")]
        public string Name { get; set; } = default!;

        [Required]
        [JsonProperty("photoUrls")]
        public List<string> PhotoUrls { get; set; } = default!;

        [JsonProperty("tags")]
        public List<Tag> Tags { get; set; }

        /// <summary>
        /// pet status in the store
        /// </summary>
        [JsonProperty("status")]
        public string Status { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public class Tag
    {
        [JsonProperty("id")]
        public long? Id { get; set; }

        [JsonProperty("name")]
        public string Name { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public class Order
    {
        [JsonProperty("id")]
        public long? Id { get; set; }

        [JsonProperty("petId")]
        public long? PetId { get; set; }

        [JsonProperty("quantity")]
        public int? Quantity { get; set; }

        [JsonProperty("shipDate")]
        public DateTime? ShipDate { get; set; }

        /// <summary>
        /// Order Status
        /// </summary>
        [JsonProperty("status")]
        public string Status { get; set; }

        [JsonProperty("complete")]
        public bool? Complete { get; set; }

    }

    /// <summary>
    /// 
    /// </summary>
    public class User
    {
        [JsonProperty("id")]
        public long? Id { get; set; }

        [JsonProperty("username")]
        public string Username { get; set; }

        [JsonProperty("firstName")]
        public string FirstName { get; set; }

        [JsonProperty("lastName")]
        public string LastName { get; set; }

        [JsonProperty("email")]
        public string Email { get; set; }

        [JsonProperty("password")]
        public string Password { get; set; }

        [JsonProperty("phone")]
        public string Phone { get; set; }

        /// <summary>
        /// User Status
        /// </summary>
        [JsonProperty("userStatus")]
        public int? UserStatus { get; set; }

    }

    /// <summary>
    /// HTTP client for PetStore API
    /// </summary>
    public class PetStoreClient
    {
        private readonly HttpClient _httpClient;
        private readonly string _baseUrl;

        /// <summary>
        /// Initializes a new instance of PetStoreClient
        /// </summary>
        /// <param name="httpClientFactory">HTTP client factory</param>
        /// <param name="baseUrl">Base URL for the API</param>
        public PetStoreClient(IHttpClientFactory httpClientFactory, string baseUrl)
        {
            _httpClient = httpClientFactory.CreateClient();
            _baseUrl = baseUrl.TrimEnd('/');
        }

        /// <summary>
        /// Initializes a new instance of PetStoreClient
        /// </summary>
        /// <param name="httpClient">HTTP client instance</param>
        /// <param name="baseUrl">Base URL for the API</param>
        public PetStoreClient(HttpClient httpClient, string baseUrl)
        {
            _httpClient = httpClient ?? throw new ArgumentNullException(nameof(httpClient));
            _baseUrl = baseUrl.TrimEnd('/');
        }

        /// <summary>
        /// uploads an image
        /// </summary>
        /// <param name="petId">ID of pet to update</param>
        /// <returns>Task<ApiResponse></returns>
        public async Task<ApiResponse> UPLOADFILE(long petId)
        {
            var url = $"/pet/{petId}/uploadImage";
            var request = new HttpRequestMessage(HttpMethod.Post, _baseUrl + url);

            var response = await _httpClient.SendAsync(request);
            response.EnsureSuccessStatusCode();

            var responseContent = await response.Content.ReadAsStringAsync();
            return JsonConvert.DeserializeObject<ApiResponse>(responseContent);
        }

        /// <summary>
        /// Add a new pet to the store
        /// </summary>
        /// <param name="body">Pet object that needs to be added to the store</param>
        /// <returns>Task</returns>
        public async Task ADDPET(Pet body)
        {
            var url = $"/pet";
            var request = new HttpRequestMessage(HttpMethod.Post, _baseUrl + url);
            var jsonContent = JsonConvert.SerializeObject(body);
            request.Content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

            var response = await _httpClient.SendAsync(request);
            response.EnsureSuccessStatusCode();

            // No return value expected
        }

        /// <summary>
        /// Update an existing pet
        /// </summary>
        /// <param name="body">Pet object that needs to be added to the store</param>
        /// <returns>Task</returns>
        public async Task UPDATEPET(Pet body)
        {
            var url = $"/pet";
            var request = new HttpRequestMessage(HttpMethod.Put, _baseUrl + url);
            var jsonContent = JsonConvert.SerializeObject(body);
            request.Content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

            var response = await _httpClient.SendAsync(request);
            response.EnsureSuccessStatusCode();

            // No return value expected
        }

        /// <summary>
        /// Finds Pets by status
        /// Multiple status values can be provided with comma separated strings
        /// </summary>
        /// <param name="status">Status values that need to be considered for filter</param>
        /// <returns>Task<List<Pet>></returns>
        public async Task<List<Pet>> FINDPETSBYSTATUS(List<object> status)
        {
            var url = $"/pet/findByStatus";
            var queryParams = new List<string>();
            queryParams.Add($"status={Uri.EscapeDataString(status.ToString())}");
            if (queryParams.Any())
                url += "?" + string.Join("&", queryParams);

            var request = new HttpRequestMessage(HttpMethod.Get, _baseUrl + url);

            var response = await _httpClient.SendAsync(request);
            response.EnsureSuccessStatusCode();

            var responseContent = await response.Content.ReadAsStringAsync();
            return JsonConvert.DeserializeObject<List<Pet>>(responseContent) ?? new List<Pet>();
        }

        /// <summary>
        /// Finds Pets by tags
        /// Multiple tags can be provided with comma separated strings. Use tag1, tag2, tag3 for testing.
        /// </summary>
        /// <param name="tags">Tags to filter by</param>
        /// <returns>Task<List<Pet>></returns>
        public async Task<List<Pet>> FINDPETSBYTAGS(List<object> tags)
        {
            var url = $"/pet/findByTags";
            var queryParams = new List<string>();
            queryParams.Add($"tags={Uri.EscapeDataString(tags.ToString())}");
            if (queryParams.Any())
                url += "?" + string.Join("&", queryParams);

            var request = new HttpRequestMessage(HttpMethod.Get, _baseUrl + url);

            var response = await _httpClient.SendAsync(request);
            response.EnsureSuccessStatusCode();

            var responseContent = await response.Content.ReadAsStringAsync();
            return JsonConvert.DeserializeObject<List<Pet>>(responseContent) ?? new List<Pet>();
        }

        /// <summary>
        /// Find pet by ID
        /// Returns a single pet
        /// </summary>
        /// <param name="petId">ID of pet to return</param>
        /// <returns>Task<Pet></returns>
        public async Task<Pet> GETPETBYID(long petId)
        {
            var url = $"/pet/{petId}";
            var request = new HttpRequestMessage(HttpMethod.Get, _baseUrl + url);

            var response = await _httpClient.SendAsync(request);
            response.EnsureSuccessStatusCode();

            var responseContent = await response.Content.ReadAsStringAsync();
            return JsonConvert.DeserializeObject<Pet>(responseContent);
        }

        /// <summary>
        /// Updates a pet in the store with form data
        /// </summary>
        /// <param name="petId">ID of pet that needs to be updated</param>
        /// <returns>Task</returns>
        public async Task UPDATEPETWITHFORM(long petId)
        {
            var url = $"/pet/{petId}";
            var request = new HttpRequestMessage(HttpMethod.Post, _baseUrl + url);

            var response = await _httpClient.SendAsync(request);
            response.EnsureSuccessStatusCode();

            // No return value expected
        }

        /// <summary>
        /// Deletes a pet
        /// </summary>
        /// <param name="petId">Pet id to delete</param>
        /// <param name="api_key"></param>
        /// <returns>Task</returns>
        public async Task DELETEPET(long petId, string? api_key = null)
        {
            var url = $"/pet/{petId}";
            var request = new HttpRequestMessage(HttpMethod.Delete, _baseUrl + url);
            if (api_key != null)
                request.Headers.Add("api_key", api_key.ToString());

            var response = await _httpClient.SendAsync(request);
            response.EnsureSuccessStatusCode();

            // No return value expected
        }

        /// <summary>
        /// Returns pet inventories by status
        /// Returns a map of status codes to quantities
        /// </summary>
        /// <returns>Task<object></returns>
        public async Task<object> GETINVENTORY()
        {
            var url = $"/store/inventory";
            var request = new HttpRequestMessage(HttpMethod.Get, _baseUrl + url);

            var response = await _httpClient.SendAsync(request);
            response.EnsureSuccessStatusCode();

            var responseContent = await response.Content.ReadAsStringAsync();
            return JsonConvert.DeserializeObject<object>(responseContent);
        }

        /// <summary>
        /// Place an order for a pet
        /// </summary>
        /// <param name="body">order placed for purchasing the pet</param>
        /// <returns>Task<Order></returns>
        public async Task<Order> PLACEORDER(Order body)
        {
            var url = $"/store/order";
            var request = new HttpRequestMessage(HttpMethod.Post, _baseUrl + url);
            var jsonContent = JsonConvert.SerializeObject(body);
            request.Content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

            var response = await _httpClient.SendAsync(request);
            response.EnsureSuccessStatusCode();

            var responseContent = await response.Content.ReadAsStringAsync();
            return JsonConvert.DeserializeObject<Order>(responseContent);
        }

        /// <summary>
        /// Find purchase order by ID
        /// For valid response try integer IDs with value >= 1 and <= 10. Other values will generated exceptions
        /// </summary>
        /// <param name="orderId">ID of pet that needs to be fetched</param>
        /// <returns>Task<Order></returns>
        public async Task<Order> GETORDERBYID(long orderId)
        {
            var url = $"/store/order/{orderId}";
            var request = new HttpRequestMessage(HttpMethod.Get, _baseUrl + url);

            var response = await _httpClient.SendAsync(request);
            response.EnsureSuccessStatusCode();

            var responseContent = await response.Content.ReadAsStringAsync();
            return JsonConvert.DeserializeObject<Order>(responseContent);
        }

        /// <summary>
        /// Delete purchase order by ID
        /// For valid response try integer IDs with positive integer value. Negative or non-integer values will generate API errors
        /// </summary>
        /// <param name="orderId">ID of the order that needs to be deleted</param>
        /// <returns>Task</returns>
        public async Task DELETEORDER(long orderId)
        {
            var url = $"/store/order/{orderId}";
            var request = new HttpRequestMessage(HttpMethod.Delete, _baseUrl + url);

            var response = await _httpClient.SendAsync(request);
            response.EnsureSuccessStatusCode();

            // No return value expected
        }

        /// <summary>
        /// Creates list of users with given input array
        /// </summary>
        /// <param name="body">List of user object</param>
        /// <returns>Task</returns>
        public async Task CREATEUSERSWITHLISTINPUT(List<User> body)
        {
            var url = $"/user/createWithList";
            var request = new HttpRequestMessage(HttpMethod.Post, _baseUrl + url);
            var jsonContent = JsonConvert.SerializeObject(body);
            request.Content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

            var response = await _httpClient.SendAsync(request);
            response.EnsureSuccessStatusCode();

            // No return value expected
        }

        /// <summary>
        /// Get user by user name
        /// </summary>
        /// <param name="username">The name that needs to be fetched. Use user1 for testing. </param>
        /// <returns>Task<User></returns>
        public async Task<User> GETUSERBYNAME(string username)
        {
            var url = $"/user/{username}";
            var request = new HttpRequestMessage(HttpMethod.Get, _baseUrl + url);

            var response = await _httpClient.SendAsync(request);
            response.EnsureSuccessStatusCode();

            var responseContent = await response.Content.ReadAsStringAsync();
            return JsonConvert.DeserializeObject<User>(responseContent);
        }

        /// <summary>
        /// Updated user
        /// This can only be done by the logged in user.
        /// </summary>
        /// <param name="username">name that need to be updated</param>
        /// <param name="body">Updated user object</param>
        /// <returns>Task</returns>
        public async Task UPDATEUSER(string username, User body)
        {
            var url = $"/user/{username}";
            var request = new HttpRequestMessage(HttpMethod.Put, _baseUrl + url);
            var jsonContent = JsonConvert.SerializeObject(body);
            request.Content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

            var response = await _httpClient.SendAsync(request);
            response.EnsureSuccessStatusCode();

            // No return value expected
        }

        /// <summary>
        /// Delete user
        /// This can only be done by the logged in user.
        /// </summary>
        /// <param name="username">The name that needs to be deleted</param>
        /// <returns>Task</returns>
        public async Task DELETEUSER(string username)
        {
            var url = $"/user/{username}";
            var request = new HttpRequestMessage(HttpMethod.Delete, _baseUrl + url);

            var response = await _httpClient.SendAsync(request);
            response.EnsureSuccessStatusCode();

            // No return value expected
        }

        /// <summary>
        /// Logs user into the system
        /// </summary>
        /// <param name="username">The user name for login</param>
        /// <param name="password">The password for login in clear text</param>
        /// <returns>Task<string></returns>
        public async Task<string> LOGINUSER(string username, string password)
        {
            var url = $"/user/login";
            var queryParams = new List<string>();
            queryParams.Add($"username={Uri.EscapeDataString(username.ToString())}");
            queryParams.Add($"password={Uri.EscapeDataString(password.ToString())}");
            if (queryParams.Any())
                url += "?" + string.Join("&", queryParams);

            var request = new HttpRequestMessage(HttpMethod.Get, _baseUrl + url);

            var response = await _httpClient.SendAsync(request);
            response.EnsureSuccessStatusCode();

            return await response.Content.ReadAsStringAsync();
        }

        /// <summary>
        /// Logs out current logged in user session
        /// </summary>
        /// <returns>Task</returns>
        public async Task LOGOUTUSER()
        {
            var url = $"/user/logout";
            var request = new HttpRequestMessage(HttpMethod.Get, _baseUrl + url);

            var response = await _httpClient.SendAsync(request);
            response.EnsureSuccessStatusCode();

            // No return value expected
        }

        /// <summary>
        /// Creates list of users with given input array
        /// </summary>
        /// <param name="body">List of user object</param>
        /// <returns>Task</returns>
        public async Task CREATEUSERSWITHARRAYINPUT(List<User> body)
        {
            var url = $"/user/createWithArray";
            var request = new HttpRequestMessage(HttpMethod.Post, _baseUrl + url);
            var jsonContent = JsonConvert.SerializeObject(body);
            request.Content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

            var response = await _httpClient.SendAsync(request);
            response.EnsureSuccessStatusCode();

            // No return value expected
        }

        /// <summary>
        /// Create user
        /// This can only be done by the logged in user.
        /// </summary>
        /// <param name="body">Created user object</param>
        /// <returns>Task</returns>
        public async Task CREATEUSER(User body)
        {
            var url = $"/user";
            var request = new HttpRequestMessage(HttpMethod.Post, _baseUrl + url);
            var jsonContent = JsonConvert.SerializeObject(body);
            request.Content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

            var response = await _httpClient.SendAsync(request);
            response.EnsureSuccessStatusCode();

            // No return value expected
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
