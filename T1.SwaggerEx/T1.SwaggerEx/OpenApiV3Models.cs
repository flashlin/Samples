using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace CodeBoyLib.Models.OpenApiV3
{
    /// <summary>
    /// Root OpenAPI 3.0 document
    /// </summary>
    public class OpenApiDocument
    {
        [JsonPropertyName("openapi")]
        public string OpenApi { get; set; } = string.Empty;

        [JsonPropertyName("info")]
        public Info Info { get; set; } = new Info();

        [JsonPropertyName("servers")]
        public List<Server>? Servers { get; set; }

        [JsonPropertyName("paths")]
        public Dictionary<string, PathItem> Paths { get; set; } = new Dictionary<string, PathItem>();

        [JsonPropertyName("components")]
        public Components? Components { get; set; }

        [JsonPropertyName("security")]
        public List<Dictionary<string, List<string>>>? Security { get; set; }

        [JsonPropertyName("tags")]
        public List<Tag>? Tags { get; set; }

        [JsonPropertyName("externalDocs")]
        public ExternalDocumentation? ExternalDocs { get; set; }
    }

    public class Info
    {
        [JsonPropertyName("title")]
        public string Title { get; set; } = string.Empty;

        [JsonPropertyName("description")]
        public string? Description { get; set; }

        [JsonPropertyName("termsOfService")]
        public string? TermsOfService { get; set; }

        [JsonPropertyName("contact")]
        public Contact? Contact { get; set; }

        [JsonPropertyName("license")]
        public License? License { get; set; }

        [JsonPropertyName("version")]
        public string Version { get; set; } = string.Empty;
    }

    public class Contact
    {
        [JsonPropertyName("name")]
        public string? Name { get; set; }

        [JsonPropertyName("url")]
        public string? Url { get; set; }

        [JsonPropertyName("email")]
        public string? Email { get; set; }
    }

    public class License
    {
        [JsonPropertyName("name")]
        public string Name { get; set; } = string.Empty;

        [JsonPropertyName("url")]
        public string? Url { get; set; }
    }

    public class Server
    {
        [JsonPropertyName("url")]
        public string Url { get; set; } = string.Empty;

        [JsonPropertyName("description")]
        public string? Description { get; set; }

        [JsonPropertyName("variables")]
        public Dictionary<string, ServerVariable>? Variables { get; set; }
    }

    public class ServerVariable
    {
        [JsonPropertyName("enum")]
        public List<string>? Enum { get; set; }

        [JsonPropertyName("default")]
        public string Default { get; set; } = string.Empty;

        [JsonPropertyName("description")]
        public string? Description { get; set; }
    }

    public class Components
    {
        [JsonPropertyName("schemas")]
        public Dictionary<string, Schema>? Schemas { get; set; }

        [JsonPropertyName("responses")]
        public Dictionary<string, Response>? Responses { get; set; }

        [JsonPropertyName("parameters")]
        public Dictionary<string, Parameter>? Parameters { get; set; }

        [JsonPropertyName("examples")]
        public Dictionary<string, Example>? Examples { get; set; }

        [JsonPropertyName("requestBodies")]
        public Dictionary<string, RequestBody>? RequestBodies { get; set; }

        [JsonPropertyName("headers")]
        public Dictionary<string, Header>? Headers { get; set; }

        [JsonPropertyName("securitySchemes")]
        public Dictionary<string, SecurityScheme>? SecuritySchemes { get; set; }

        [JsonPropertyName("links")]
        public Dictionary<string, Link>? Links { get; set; }

        [JsonPropertyName("callbacks")]
        public Dictionary<string, Callback>? Callbacks { get; set; }
    }

    public class PathItem
    {
        [JsonPropertyName("$ref")]
        public string? Ref { get; set; }

        [JsonPropertyName("summary")]
        public string? Summary { get; set; }

        [JsonPropertyName("description")]
        public string? Description { get; set; }

        [JsonPropertyName("get")]
        public Operation? Get { get; set; }

        [JsonPropertyName("put")]
        public Operation? Put { get; set; }

        [JsonPropertyName("post")]
        public Operation? Post { get; set; }

        [JsonPropertyName("delete")]
        public Operation? Delete { get; set; }

        [JsonPropertyName("options")]
        public Operation? Options { get; set; }

        [JsonPropertyName("head")]
        public Operation? Head { get; set; }

        [JsonPropertyName("patch")]
        public Operation? Patch { get; set; }

        [JsonPropertyName("trace")]
        public Operation? Trace { get; set; }

        [JsonPropertyName("servers")]
        public List<Server>? Servers { get; set; }

        [JsonPropertyName("parameters")]
        public List<Parameter>? Parameters { get; set; }
    }

    public class Operation
    {
        [JsonPropertyName("tags")]
        public List<string>? Tags { get; set; }

        [JsonPropertyName("summary")]
        public string? Summary { get; set; }

        [JsonPropertyName("description")]
        public string? Description { get; set; }

        [JsonPropertyName("externalDocs")]
        public ExternalDocumentation? ExternalDocs { get; set; }

        [JsonPropertyName("operationId")]
        public string? OperationId { get; set; }

        [JsonPropertyName("parameters")]
        public List<Parameter>? Parameters { get; set; }

        [JsonPropertyName("requestBody")]
        public RequestBody? RequestBody { get; set; }

        [JsonPropertyName("responses")]
        public Dictionary<string, Response> Responses { get; set; } = new Dictionary<string, Response>();

        [JsonPropertyName("callbacks")]
        public Dictionary<string, Callback>? Callbacks { get; set; }

        [JsonPropertyName("deprecated")]
        public bool Deprecated { get; set; }

        [JsonPropertyName("security")]
        public List<Dictionary<string, List<string>>>? Security { get; set; }

        [JsonPropertyName("servers")]
        public List<Server>? Servers { get; set; }
    }

    public class Parameter
    {
        [JsonPropertyName("name")]
        public string Name { get; set; } = string.Empty;

        [JsonPropertyName("in")]
        public string In { get; set; } = string.Empty;

        [JsonPropertyName("description")]
        public string? Description { get; set; }

        [JsonPropertyName("required")]
        public bool Required { get; set; }

        [JsonPropertyName("deprecated")]
        public bool Deprecated { get; set; }

        [JsonPropertyName("allowEmptyValue")]
        public bool AllowEmptyValue { get; set; }

        [JsonPropertyName("style")]
        public string? Style { get; set; }

        [JsonPropertyName("explode")]
        public bool Explode { get; set; }

        [JsonPropertyName("allowReserved")]
        public bool AllowReserved { get; set; }

        [JsonPropertyName("schema")]
        public Schema? Schema { get; set; }

        [JsonPropertyName("example")]
        public object? Example { get; set; }

        [JsonPropertyName("examples")]
        public Dictionary<string, Example>? Examples { get; set; }

        [JsonPropertyName("content")]
        public Dictionary<string, MediaType>? Content { get; set; }
    }

    public class RequestBody
    {
        [JsonPropertyName("description")]
        public string? Description { get; set; }

        [JsonPropertyName("content")]
        public Dictionary<string, MediaType> Content { get; set; } = new Dictionary<string, MediaType>();

        [JsonPropertyName("required")]
        public bool Required { get; set; }
    }

    public class MediaType
    {
        [JsonPropertyName("schema")]
        public Schema? Schema { get; set; }

        [JsonPropertyName("example")]
        public object? Example { get; set; }

        [JsonPropertyName("examples")]
        public Dictionary<string, Example>? Examples { get; set; }

        [JsonPropertyName("encoding")]
        public Dictionary<string, Encoding>? Encoding { get; set; }
    }

    public class Encoding
    {
        [JsonPropertyName("contentType")]
        public string? ContentType { get; set; }

        [JsonPropertyName("headers")]
        public Dictionary<string, Header>? Headers { get; set; }

        [JsonPropertyName("style")]
        public string? Style { get; set; }

        [JsonPropertyName("explode")]
        public bool Explode { get; set; }

        [JsonPropertyName("allowReserved")]
        public bool AllowReserved { get; set; }
    }

    public class Response
    {
        [JsonPropertyName("description")]
        public string Description { get; set; } = string.Empty;

        [JsonPropertyName("headers")]
        public Dictionary<string, Header>? Headers { get; set; }

        [JsonPropertyName("content")]
        public Dictionary<string, MediaType>? Content { get; set; }

        [JsonPropertyName("links")]
        public Dictionary<string, Link>? Links { get; set; }
    }

    public class Header
    {
        [JsonPropertyName("description")]
        public string? Description { get; set; }

        [JsonPropertyName("required")]
        public bool Required { get; set; }

        [JsonPropertyName("deprecated")]
        public bool Deprecated { get; set; }

        [JsonPropertyName("allowEmptyValue")]
        public bool AllowEmptyValue { get; set; }

        [JsonPropertyName("style")]
        public string? Style { get; set; }

        [JsonPropertyName("explode")]
        public bool Explode { get; set; }

        [JsonPropertyName("allowReserved")]
        public bool AllowReserved { get; set; }

        [JsonPropertyName("schema")]
        public Schema? Schema { get; set; }

        [JsonPropertyName("example")]
        public object? Example { get; set; }

        [JsonPropertyName("examples")]
        public Dictionary<string, Example>? Examples { get; set; }

        [JsonPropertyName("content")]
        public Dictionary<string, MediaType>? Content { get; set; }
    }

    public class Schema
    {
        [JsonPropertyName("title")]
        public string? Title { get; set; }

        [JsonPropertyName("multipleOf")]
        public decimal? MultipleOf { get; set; }

        [JsonPropertyName("maximum")]
        public decimal? Maximum { get; set; }

        [JsonPropertyName("exclusiveMaximum")]
        public bool ExclusiveMaximum { get; set; }

        [JsonPropertyName("minimum")]
        public decimal? Minimum { get; set; }

        [JsonPropertyName("exclusiveMinimum")]
        public bool ExclusiveMinimum { get; set; }

        [JsonPropertyName("maxLength")]
        public int? MaxLength { get; set; }

        [JsonPropertyName("minLength")]
        public int? MinLength { get; set; }

        [JsonPropertyName("pattern")]
        public string? Pattern { get; set; }

        [JsonPropertyName("maxItems")]
        public int? MaxItems { get; set; }

        [JsonPropertyName("minItems")]
        public int? MinItems { get; set; }

        [JsonPropertyName("uniqueItems")]
        public bool UniqueItems { get; set; }

        [JsonPropertyName("maxProperties")]
        public int? MaxProperties { get; set; }

        [JsonPropertyName("minProperties")]
        public int? MinProperties { get; set; }

        [JsonPropertyName("required")]
        public List<string>? Required { get; set; }

        [JsonPropertyName("enum")]
        public List<object>? Enum { get; set; }

        [JsonPropertyName("type")]
        public string? Type { get; set; }

        [JsonPropertyName("allOf")]
        public List<Schema>? AllOf { get; set; }

        [JsonPropertyName("oneOf")]
        public List<Schema>? OneOf { get; set; }

        [JsonPropertyName("anyOf")]
        public List<Schema>? AnyOf { get; set; }

        [JsonPropertyName("not")]
        public Schema? Not { get; set; }

        [JsonPropertyName("items")]
        public Schema? Items { get; set; }

        [JsonPropertyName("properties")]
        public Dictionary<string, Schema>? Properties { get; set; }

        [JsonPropertyName("additionalProperties")]
        public object? AdditionalProperties { get; set; }

        [JsonPropertyName("description")]
        public string? Description { get; set; }

        [JsonPropertyName("format")]
        public string? Format { get; set; }

        [JsonPropertyName("default")]
        public object? Default { get; set; }

        [JsonPropertyName("nullable")]
        public bool Nullable { get; set; }

        [JsonPropertyName("discriminator")]
        public Discriminator? Discriminator { get; set; }

        [JsonPropertyName("readOnly")]
        public bool ReadOnly { get; set; }

        [JsonPropertyName("writeOnly")]
        public bool WriteOnly { get; set; }

        [JsonPropertyName("xml")]
        public Xml? Xml { get; set; }

        [JsonPropertyName("externalDocs")]
        public ExternalDocumentation? ExternalDocs { get; set; }

        [JsonPropertyName("example")]
        public object? Example { get; set; }

        [JsonPropertyName("deprecated")]
        public bool Deprecated { get; set; }

        [JsonPropertyName("$ref")]
        public string? Ref { get; set; }
    }

    public class Discriminator
    {
        [JsonPropertyName("propertyName")]
        public string PropertyName { get; set; } = string.Empty;

        [JsonPropertyName("mapping")]
        public Dictionary<string, string>? Mapping { get; set; }
    }

    public class Xml
    {
        [JsonPropertyName("name")]
        public string? Name { get; set; }

        [JsonPropertyName("namespace")]
        public string? Namespace { get; set; }

        [JsonPropertyName("prefix")]
        public string? Prefix { get; set; }

        [JsonPropertyName("attribute")]
        public bool Attribute { get; set; }

        [JsonPropertyName("wrapped")]
        public bool Wrapped { get; set; }
    }

    public class Example
    {
        [JsonPropertyName("summary")]
        public string? Summary { get; set; }

        [JsonPropertyName("description")]
        public string? Description { get; set; }

        [JsonPropertyName("value")]
        public object? Value { get; set; }

        [JsonPropertyName("externalValue")]
        public string? ExternalValue { get; set; }
    }

    public class Link
    {
        [JsonPropertyName("operationRef")]
        public string? OperationRef { get; set; }

        [JsonPropertyName("operationId")]
        public string? OperationId { get; set; }

        [JsonPropertyName("parameters")]
        public Dictionary<string, object>? Parameters { get; set; }

        [JsonPropertyName("requestBody")]
        public object? RequestBody { get; set; }

        [JsonPropertyName("description")]
        public string? Description { get; set; }

        [JsonPropertyName("server")]
        public Server? Server { get; set; }
    }

    public class Callback
    {
        // Callback uses expression as key, PathItem as value
        // This is a simplified representation
        public Dictionary<string, PathItem> Expressions { get; set; } = new Dictionary<string, PathItem>();
    }

    public class SecurityScheme
    {
        [JsonPropertyName("type")]
        public string Type { get; set; } = string.Empty;

        [JsonPropertyName("description")]
        public string? Description { get; set; }

        [JsonPropertyName("name")]
        public string? Name { get; set; }

        [JsonPropertyName("in")]
        public string? In { get; set; }

        [JsonPropertyName("scheme")]
        public string? Scheme { get; set; }

        [JsonPropertyName("bearerFormat")]
        public string? BearerFormat { get; set; }

        [JsonPropertyName("flows")]
        public OAuthFlows? Flows { get; set; }

        [JsonPropertyName("openIdConnectUrl")]
        public string? OpenIdConnectUrl { get; set; }
    }

    public class OAuthFlows
    {
        [JsonPropertyName("implicit")]
        public OAuthFlow? Implicit { get; set; }

        [JsonPropertyName("password")]
        public OAuthFlow? Password { get; set; }

        [JsonPropertyName("clientCredentials")]
        public OAuthFlow? ClientCredentials { get; set; }

        [JsonPropertyName("authorizationCode")]
        public OAuthFlow? AuthorizationCode { get; set; }
    }

    public class OAuthFlow
    {
        [JsonPropertyName("authorizationUrl")]
        public string? AuthorizationUrl { get; set; }

        [JsonPropertyName("tokenUrl")]
        public string? TokenUrl { get; set; }

        [JsonPropertyName("refreshUrl")]
        public string? RefreshUrl { get; set; }

        [JsonPropertyName("scopes")]
        public Dictionary<string, string> Scopes { get; set; } = new Dictionary<string, string>();
    }

    public class Tag
    {
        [JsonPropertyName("name")]
        public string Name { get; set; } = string.Empty;

        [JsonPropertyName("description")]
        public string? Description { get; set; }

        [JsonPropertyName("externalDocs")]
        public ExternalDocumentation? ExternalDocs { get; set; }
    }

    public class ExternalDocumentation
    {
        [JsonPropertyName("description")]
        public string? Description { get; set; }

        [JsonPropertyName("url")]
        public string Url { get; set; } = string.Empty;
    }
}
