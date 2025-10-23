using System;
using System.Text.RegularExpressions;
using T1.EfCodeFirstGenerateCli.Models;

namespace T1.EfCodeFirstGenerateCli.ConfigParser
{
    public static class MermaidRelationshipParser
    {
        // Regex pattern components to match Mermaid relationship syntax
        // Format: PrincipalEntity ||--o{ DependentEntity : "Principal.Key = Dependent.ForeignKey"
        private const string EntityNamePattern = @"(\w+)";                    // Captures entity name (e.g., User, Order)
        private const string RelationshipSymbolPattern = @"([\|\}o\{\-\>]+)"; // Captures symbols (e.g., ||--o{, ||-->||)
        private const string KeyMappingPattern = @"""([^""]+)""";             // Captures quoted key mapping
        
        private static readonly Regex RelationshipPattern = new Regex(
            @$"^\s*{EntityNamePattern}\s+{RelationshipSymbolPattern}\s+{EntityNamePattern}\s*:\s*{KeyMappingPattern}",
            RegexOptions.Compiled
        );

        public static EntityRelationship? ParseRelationship(string line)
        {
            var match = RelationshipPattern.Match(line);
            if (!match.Success)
            {
                return null;
            }

            var principalEntity = match.Groups[1].Value;
            var relationshipSymbol = match.Groups[2].Value;
            var dependentEntity = match.Groups[3].Value;
            var keyMapping = match.Groups[4].Value;

            // Parse relationship type and navigation type
            var (relType, navType, isPrincipalOptional, isDependentOptional) = ParseRelationshipSymbol(relationshipSymbol);
            if (!relType.HasValue)
            {
                return null;
            }

            // Parse key mapping: "Principal.Key = Dependent.ForeignKey"
            var keys = ParseKeyMapping(keyMapping);
            if (keys == null)
            {
                return null;
            }

            return new EntityRelationship
            {
                PrincipalEntity = principalEntity,
                PrincipalKey = keys.Value.PrincipalKey,
                DependentEntity = dependentEntity,
                ForeignKey = keys.Value.ForeignKey,
                Type = relType.Value,
                NavigationType = navType,
                IsPrincipalOptional = isPrincipalOptional,
                IsDependentOptional = isDependentOptional
            };
        }

        private static (RelationshipType?, NavigationType, bool IsPrincipalOptional, bool IsDependentOptional) ParseRelationshipSymbol(string symbol)
        {
            // Remove all spaces for easier matching
            var cleaned = symbol.Replace(" ", "");

            return cleaned switch
            {
                // Existing patterns
                "||--o{" => (RelationshipType.OneToMany, NavigationType.Bidirectional, false, false),
                "||--||" => (RelationshipType.OneToOne, NavigationType.Bidirectional, false, false),
                "||-->o{" => (RelationshipType.OneToMany, NavigationType.Unidirectional, false, false),
                "||-->||" => (RelationshipType.OneToOne, NavigationType.Unidirectional, false, false),
                "o{--||" => (RelationshipType.ManyToOne, NavigationType.Bidirectional, false, false),
                "o{-->||" => (RelationshipType.ManyToOne, NavigationType.Unidirectional, false, false),
                
                // New patterns with zero-or-one
                "||--o|" => (RelationshipType.OneToOne, NavigationType.Bidirectional, false, true),   // dependent optional
                "||-->o|" => (RelationshipType.OneToOne, NavigationType.Unidirectional, false, true), // dependent optional
                "o|--||" => (RelationshipType.OneToOne, NavigationType.Bidirectional, true, false),   // principal optional
                "o|-->||" => (RelationshipType.OneToOne, NavigationType.Unidirectional, true, false), // principal optional
                
                _ => (null, NavigationType.Bidirectional, false, false)
            };
        }

        private static (string PrincipalKey, string ForeignKey)? ParseKeyMapping(string mapping)
        {
            // Expected format: "Principal.Key = Dependent.ForeignKey"
            var parts = mapping.Split('=');
            if (parts.Length != 2)
            {
                return null;
            }

            var leftParts = parts[0].Trim().Split('.');
            var rightParts = parts[1].Trim().Split('.');

            if (leftParts.Length != 2 || rightParts.Length != 2)
            {
                return null;
            }

            return (leftParts[1], rightParts[1]);
        }
    }
}

