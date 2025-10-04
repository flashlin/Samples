using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using CodeBoyLib.Models;
using T1.Standard.IO;

namespace CodeBoyLib.Services
{
    /// <summary>
    /// Service for generating TypeScript API client code from Swagger/OpenAPI specifications
    /// </summary>
    public class SwaggerClientTypescriptCodeGenerator
    {
        /// <summary>
        /// Generates TypeScript API client code for the specified API
        /// </summary>
        /// <param name="apiName">Name of the API (used for naming the exported object)</param>
        /// <param name="apiInfo">Swagger API information containing endpoints and models</param>
        /// <returns>Generated TypeScript code as string</returns>
        public string Generate(string apiName, SwaggerApiInfo apiInfo)
        {
            var output = new IndentStringBuilder();
            
            // Add import statement
            output.WriteLine("import request from './request';");
            output.WriteLine();
            
            // Generate interfaces for request/response models
            GenerateInterfaces(output, apiInfo);
            
            // Generate API client object
            GenerateApiClient(output, apiName, apiInfo);
            
            return output.ToString();
        }

        /// <summary>
        /// Generates TypeScript interfaces for all class definitions used in the API
        /// </summary>
        /// <param name="output">String builder for output</param>
        /// <param name="apiInfo">API information containing class definitions</param>
        private void GenerateInterfaces(IndentStringBuilder output, SwaggerApiInfo apiInfo)
        {
            if (!apiInfo.ClassDefinitions.Any())
                return;

            foreach (var classDefinition in apiInfo.ClassDefinitions.Values)
            {
                GenerateInterface(output, classDefinition);
                output.WriteLine();
            }
        }

        /// <summary>
        /// Generates a single TypeScript interface from a class definition
        /// </summary>
        /// <param name="output">String builder for output</param>
        /// <param name="classDefinition">Class definition to generate interface for</param>
        private void GenerateInterface(IndentStringBuilder output, ClassDefinition classDefinition)
        {
            // Add interface documentation if description is available
            if (!string.IsNullOrWhiteSpace(classDefinition.Description))
            {
                output.WriteLine("/**");
                output.WriteLine($" * {classDefinition.Description}");
                output.WriteLine(" */");
            }

            // Handle enums differently
            if (classDefinition.IsEnum)
            {
                GenerateEnum(output, classDefinition);
                return;
            }

            output.WriteLine($"export interface {classDefinition.Name} {{");
            output.Indent++;

            if (classDefinition.Properties.Any())
            {
                foreach (var property in classDefinition.Properties)
                {
                    GenerateInterfaceProperty(output, property);
                }
            }

            output.Indent--;
            output.WriteLine("}");
        }

        /// <summary>
        /// Generates a TypeScript enum from a class definition
        /// </summary>
        /// <param name="output">String builder for output</param>
        /// <param name="classDefinition">Class definition representing an enum</param>
        private void GenerateEnum(IndentStringBuilder output, ClassDefinition classDefinition)
        {
            output.WriteLine($"export enum {classDefinition.Name} {{");
            output.Indent++;

            if (classDefinition.EnumValues.Any())
            {
                for (int i = 0; i < classDefinition.EnumValues.Count; i++)
                {
                    var enumValue = classDefinition.EnumValues[i];
                    var enumName = enumValue.Replace(" ", "").Replace("-", "_");
                    
                    if (classDefinition.IsNumericEnum)
                    {
                        output.Write($"{enumName} = {enumValue}");
                    }
                    else
                    {
                        output.Write($"{enumName} = '{enumValue}'");
                    }

                    if (i < classDefinition.EnumValues.Count - 1)
                    {
                        output.Write(",");
                    }
                    
                    output.WriteLine();
                }
            }

            output.Indent--;
            output.WriteLine("}");
        }

        /// <summary>
        /// Generates a TypeScript interface property
        /// </summary>
        /// <param name="output">String builder for output</param>
        /// <param name="property">Property to generate</param>
        private void GenerateInterfaceProperty(IndentStringBuilder output, ClassProperty property)
        {
            var propertyName = ToCamelCase(property.Name);
            var propertyType = ConvertToTypeScriptType(property.Type, property.Format, false);
            var optional = property.IsRequired ? "" : "?";

            output.WriteLine($"{propertyName}{optional}: {propertyType};");
        }

        /// <summary>
        /// Generates the main API client object with all endpoint methods
        /// </summary>
        /// <param name="output">String builder for output</param>
        /// <param name="apiName">Name of the API</param>
        /// <param name="apiInfo">API information containing endpoints</param>
        private void GenerateApiClient(IndentStringBuilder output, string apiName, SwaggerApiInfo apiInfo)
        {
            output.WriteLine($"export const {ToCamelCase(apiName)}Api = {{");
            output.Indent++;

            if (apiInfo.Endpoints.Any())
            {
                for (int i = 0; i < apiInfo.Endpoints.Count; i++)
                {
                    var endpoint = apiInfo.Endpoints[i];
                    GenerateEndpointMethod(output, endpoint);
                    
                    // Add comma if not the last endpoint
                    if (i < apiInfo.Endpoints.Count - 1)
                    {
                        output.Write(",");
                    }
                    
                    output.WriteLine();
                }
            }

            output.Indent--;
            output.WriteLine("};");
        }

        /// <summary>
        /// Generates a single endpoint method for the API client
        /// </summary>
        /// <param name="output">String builder for output</param>
        /// <param name="endpoint">Endpoint information</param>
        private void GenerateEndpointMethod(IndentStringBuilder output, SwaggerEndpoint endpoint)
        {
            var methodName = ToCamelCase(endpoint.OperationId);
            var parameters = GenerateMethodParameters(endpoint);
            var returnType = GetReturnType(endpoint);
            var httpMethod = endpoint.HttpMethod.ToLowerCase();

            // Generate method signature
            output.Write($"{methodName}({parameters}): Promise<{returnType}> {{");
            output.WriteLine();
            output.Indent++;

            // Generate method body
            GenerateMethodBody(output, endpoint, httpMethod);

            output.Indent--;
            output.Write("}");
        }

        /// <summary>
        /// Generates the method parameters string for an endpoint
        /// </summary>
        /// <param name="endpoint">Endpoint information</param>
        /// <returns>Parameters string for the method signature</returns>
        private string GenerateMethodParameters(SwaggerEndpoint endpoint)
        {
            var parameters = new List<string>();

            // Add path parameters
            foreach (var param in endpoint.Parameters.Where(p => p.Location == "path"))
            {
                var paramType = ConvertToTypeScriptType(param.Type, "", false);
                parameters.Add($"{ToCamelCase(param.Name)}: {paramType}");
            }

            // Add query parameters as an optional object
            var queryParams = endpoint.Parameters.Where(p => p.Location == "query").ToList();
            if (queryParams.Any())
            {
                parameters.Add("params?: { [key: string]: any }");
            }

            // Add body parameters
            var bodyParams = endpoint.Parameters.Where(p => p.Location == "body").ToList();
            if (bodyParams.Any())
            {
                var bodyParam = bodyParams.First();
                parameters.Add($"data: {bodyParam.Type}");
            }

            return string.Join(", ", parameters);
        }

        /// <summary>
        /// Gets the return type for an endpoint method
        /// </summary>
        /// <param name="endpoint">Endpoint information</param>
        /// <returns>TypeScript return type</returns>
        private string GetReturnType(SwaggerEndpoint endpoint)
        {
            if (!string.IsNullOrWhiteSpace(endpoint.ResponseType.Type))
            {
                return endpoint.ResponseType.IsArray ? $"{endpoint.ResponseType.Type}[]" : endpoint.ResponseType.Type;
            }

            return "any";
        }

        /// <summary>
        /// Generates the method body for an endpoint
        /// </summary>
        /// <param name="output">String builder for output</param>
        /// <param name="endpoint">Endpoint information</param>
        /// <param name="httpMethod">HTTP method (get, post, put, delete, etc.)</param>
        private void GenerateMethodBody(IndentStringBuilder output, SwaggerEndpoint endpoint, string httpMethod)
        {
            var url = ProcessUrlPath(endpoint.Path);
            var hasBody = endpoint.Parameters.Any(p => p.Location == "body");
            var hasQueryParams = endpoint.Parameters.Any(p => p.Location == "query");

            if (httpMethod == "get" || httpMethod == "delete")
            {
                if (hasQueryParams)
                {
                    output.WriteLine($"return request.{httpMethod}(`{url}`, {{ params }});");
                }
                else
                {
                    output.WriteLine($"return request.{httpMethod}(`{url}`);");
                }
            }
            else if (httpMethod == "post" || httpMethod == "put" || httpMethod == "patch")
            {
                if (hasBody && hasQueryParams)
                {
                    output.WriteLine($"return request.{httpMethod}(`{url}`, data, {{ params }});");
                }
                else if (hasBody)
                {
                    output.WriteLine($"return request.{httpMethod}(`{url}`, data);");
                }
                else if (hasQueryParams)
                {
                    output.WriteLine($"return request.{httpMethod}(`{url}`, {{}}, {{ params }});");
                }
                else
                {
                    output.WriteLine($"return request.{httpMethod}(`{url}`);");
                }
            }
            else
            {
                // Fallback for other HTTP methods
                output.WriteLine($"return request.{httpMethod}(`{url}`);");
            }
        }

        /// <summary>
        /// Processes URL path to handle path parameters
        /// </summary>
        /// <param name="path">Original path with parameters like {id}</param>
        /// <returns>Processed path for TypeScript template literals</returns>
        private string ProcessUrlPath(string path)
        {
            // Convert {id} to ${id} for template literals
            return path.Replace("{", "${");
        }

        /// <summary>
        /// Converts a Swagger/OpenAPI type to TypeScript type
        /// </summary>
        /// <param name="swaggerType">Swagger type</param>
        /// <param name="format">Type format</param>
        /// <param name="isArray">Whether the type is an array</param>
        /// <returns>TypeScript type string</returns>
        private string ConvertToTypeScriptType(string swaggerType, string? format, bool isArray)
        {
            if (string.IsNullOrEmpty(swaggerType))
                return "any";

            if (swaggerType.StartsWith("List<") && swaggerType.EndsWith(">"))
            {
                var innerType = swaggerType.Substring(5, swaggerType.Length - 6);
                var convertedInnerType = ConvertToTypeScriptType(innerType, format, false);
                return $"{convertedInnerType}[]";
            }

            var baseType = swaggerType?.ToLower() switch
            {
                "string" => format?.ToLower() switch
                {
                    "date" => "string",
                    "date-time" => "string",
                    "uuid" => "string",
                    _ => "string"
                },
                "integer" => "number",
                "int" => "number",
                "long" => "number",
                "number" => "number",
                "float" => "number",
                "double" => "number",
                "decimal" => "number",
                "boolean" => "boolean",
                "bool" => "boolean",
                "object" => "any",
                "array" => "any[]",
                null or "" => "any",
                _ => swaggerType
            };

            return isArray ? $"{baseType}[]" : baseType;
        }

        /// <summary>
        /// Converts PascalCase string to camelCase
        /// </summary>
        /// <param name="input">Input string in PascalCase</param>
        /// <returns>String in camelCase</returns>
        private string ToCamelCase(string input)
        {
            if (string.IsNullOrWhiteSpace(input))
                return input;

            if (input.Length == 1)
                return input.ToLower();

            return char.ToLower(input[0]) + input.Substring(1);
        }
    }

    /// <summary>
    /// Extension methods for string operations
    /// </summary>
    public static class StringExtensions
    {
        /// <summary>
        /// Converts string to lowercase
        /// </summary>
        /// <param name="input">Input string</param>
        /// <returns>Lowercase string</returns>
        public static string ToLowerCase(this string input)
        {
            return input?.ToLower() ?? string.Empty;
        }
    }
}
