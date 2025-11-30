using System.Collections.Generic;

namespace T1.SwaggerEx.Models
{
    // Represents a single parameter for an endpoint
    public class EndpointParameter
    {
        public string Name { get; set; } = string.Empty;
        public string Type { get; set; } = string.Empty;
        public bool IsRequired { get; set; }
        public string Location { get; set; } = string.Empty; // query, path, body, header
        public string Description { get; set; } = string.Empty;
    }

    // Represents a property of a class
    public class ClassProperty
    {
        public string Name { get; set; } = string.Empty;
        public string Type { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public bool IsRequired { get; set; }
        public bool IsNullable { get; set; }
        public object? DefaultValue { get; set; }
        public string Format { get; set; } = string.Empty;
    }

    // Represents a complete class definition
    public class ClassDefinition
    {
        public string Name { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public List<ClassProperty> Properties { get; set; } = new List<ClassProperty>();
        public List<string> RequiredProperties { get; set; } = new List<string>();
        public string Type { get; set; } = "object"; // object, array, etc.
        public bool IsEnum { get; set; }
        public List<string> EnumValues { get; set; } = new List<string>();
        public bool IsNumericEnum { get; set; }
    }

    // Represents a property of a response type
    public class ResponseProperty
    {
        public string Name { get; set; } = string.Empty;
        public string Type { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
    }

    // Represents the response type structure
    public class ResponseType
    {
        public string Type { get; set; } = string.Empty;
        public List<ResponseProperty> Properties { get; set; } = new List<ResponseProperty>();
        public bool IsArray { get; set; }
        public string Description { get; set; } = string.Empty;
    }

    // Represents a complete Swagger endpoint
    public class SwaggerEndpoint
    {
        public string Path { get; set; } = string.Empty;
        public string HttpMethod { get; set; } = string.Empty;
        public string OperationId { get; set; } = string.Empty;
        public string Summary { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public List<EndpointParameter> Parameters { get; set; } = new List<EndpointParameter>();
        public ResponseType ResponseType { get; set; } = new ResponseType();
        public List<string> Tags { get; set; } = new List<string>();
        public List<string> Consumes { get; set; } = new List<string>();
        public List<string> Produces { get; set; } = new List<string>();
    }

    // Main container for all parsed Swagger information
    public class SwaggerApiInfo
    {
        public List<SwaggerEndpoint> Endpoints { get; set; } = new List<SwaggerEndpoint>();
        public Dictionary<string, ClassDefinition> ClassDefinitions { get; set; } = new Dictionary<string, ClassDefinition>();
        public string Title { get; set; } = string.Empty;
        public string Version { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public string BaseUrl { get; set; } = string.Empty;
    }
}
