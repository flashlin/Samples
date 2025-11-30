using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace CodeBoyLib.Models.SwaggerV2
{
    /// <summary>
    /// Root Swagger 2.0 document
    /// </summary>
    public class SwaggerDocument
    {
        [JsonPropertyName("swagger")]
        public string Swagger { get; set; } = string.Empty;

        [JsonPropertyName("info")]
        public Info Info { get; set; } = new Info();

        [JsonPropertyName("host")]
        public string? Host { get; set; }

        [JsonPropertyName("basePath")]
        public string? BasePath { get; set; }

        [JsonPropertyName("schemes")]
        public List<string> Schemes { get; set; } = new List<string>();

        [JsonPropertyName("consumes")]
        public List<string> Consumes { get; set; } = new List<string>();

        [JsonPropertyName("produces")]
        public List<string> Produces { get; set; } = new List<string>();

        [JsonPropertyName("paths")]
        public Dictionary<string, PathItem> Paths { get; set; } = new Dictionary<string, PathItem>();

        [JsonPropertyName("definitions")]
        public Dictionary<string, Schema> Definitions { get; set; } = new Dictionary<string, Schema>();

        [JsonPropertyName("parameters")]
        public Dictionary<string, Parameter>? Parameters { get; set; }

        [JsonPropertyName("responses")]
        public Dictionary<string, Response>? Responses { get; set; }

        [JsonPropertyName("securityDefinitions")]
        public Dictionary<string, SecurityScheme>? SecurityDefinitions { get; set; }

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

    public class PathItem
    {
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

        [JsonPropertyName("consumes")]
        public List<string>? Consumes { get; set; }

        [JsonPropertyName("produces")]
        public List<string>? Produces { get; set; }

        [JsonPropertyName("parameters")]
        public List<Parameter>? Parameters { get; set; }

        [JsonPropertyName("responses")]
        public Dictionary<string, Response> Responses { get; set; } = new Dictionary<string, Response>();

        [JsonPropertyName("schemes")]
        public List<string>? Schemes { get; set; }

        [JsonPropertyName("deprecated")]
        public bool Deprecated { get; set; }

        [JsonPropertyName("security")]
        public List<Dictionary<string, List<string>>>? Security { get; set; }
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

        // For non-body parameters
        [JsonPropertyName("type")]
        public string? Type { get; set; }

        [JsonPropertyName("format")]
        public string? Format { get; set; }

        [JsonPropertyName("allowEmptyValue")]
        public bool AllowEmptyValue { get; set; }

        [JsonPropertyName("items")]
        public Items? Items { get; set; }

        [JsonPropertyName("collectionFormat")]
        public string? CollectionFormat { get; set; }

        [JsonPropertyName("default")]
        public object? Default { get; set; }

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

        [JsonPropertyName("enum")]
        public List<object>? Enum { get; set; }

        [JsonPropertyName("multipleOf")]
        public decimal? MultipleOf { get; set; }

        // For body parameters
        [JsonPropertyName("schema")]
        public Schema? Schema { get; set; }
    }

    public class Items
    {
        [JsonPropertyName("type")]
        public string Type { get; set; } = string.Empty;

        [JsonPropertyName("format")]
        public string? Format { get; set; }

        [JsonPropertyName("items")]
        public Items? NestedItems { get; set; }

        [JsonPropertyName("collectionFormat")]
        public string? CollectionFormat { get; set; }

        [JsonPropertyName("default")]
        public object? Default { get; set; }

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

        [JsonPropertyName("enum")]
        public List<object>? Enum { get; set; }

        [JsonPropertyName("multipleOf")]
        public decimal? MultipleOf { get; set; }
    }

    public class Response
    {
        [JsonPropertyName("description")]
        public string Description { get; set; } = string.Empty;

        [JsonPropertyName("schema")]
        public Schema? Schema { get; set; }

        [JsonPropertyName("headers")]
        public Dictionary<string, Header>? Headers { get; set; }

        [JsonPropertyName("examples")]
        public Dictionary<string, object>? Examples { get; set; }
    }

    public class Header
    {
        [JsonPropertyName("description")]
        public string? Description { get; set; }

        [JsonPropertyName("type")]
        public string Type { get; set; } = string.Empty;

        [JsonPropertyName("format")]
        public string? Format { get; set; }

        [JsonPropertyName("items")]
        public Items? Items { get; set; }

        [JsonPropertyName("collectionFormat")]
        public string? CollectionFormat { get; set; }

        [JsonPropertyName("default")]
        public object? Default { get; set; }

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

        [JsonPropertyName("enum")]
        public List<object>? Enum { get; set; }

        [JsonPropertyName("multipleOf")]
        public decimal? MultipleOf { get; set; }
    }

    public class Schema
    {
        [JsonPropertyName("$ref")]
        public string? Ref { get; set; }

        [JsonPropertyName("format")]
        public string? Format { get; set; }

        [JsonPropertyName("title")]
        public string? Title { get; set; }

        [JsonPropertyName("description")]
        public string? Description { get; set; }

        [JsonPropertyName("default")]
        public object? Default { get; set; }

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

        [JsonPropertyName("items")]
        public Schema? Items { get; set; }

        [JsonPropertyName("allOf")]
        public List<Schema>? AllOf { get; set; }

        [JsonPropertyName("properties")]
        public Dictionary<string, Schema>? Properties { get; set; }

        [JsonPropertyName("additionalProperties")]
        public object? AdditionalProperties { get; set; }

        [JsonPropertyName("discriminator")]
        public string? Discriminator { get; set; }

        [JsonPropertyName("readOnly")]
        public bool ReadOnly { get; set; }

        [JsonPropertyName("xml")]
        public Xml? Xml { get; set; }

        [JsonPropertyName("externalDocs")]
        public ExternalDocumentation? ExternalDocs { get; set; }

        [JsonPropertyName("example")]
        public object? Example { get; set; }
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

        [JsonPropertyName("flow")]
        public string? Flow { get; set; }

        [JsonPropertyName("authorizationUrl")]
        public string? AuthorizationUrl { get; set; }

        [JsonPropertyName("tokenUrl")]
        public string? TokenUrl { get; set; }

        [JsonPropertyName("scopes")]
        public Dictionary<string, string>? Scopes { get; set; }
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
