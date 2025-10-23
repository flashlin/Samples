using System;
using System.Text.RegularExpressions;
using T1.EfCodeFirstGenerateCli.Models;

namespace T1.EfCodeFirstGenerateCli.ConfigParser
{
    public static class MermaidRelationshipParser
    {
        // Regex pattern to match Mermaid relationship syntax
        // Format: PrincipalEntity ||--o{ DependentEntity : "Principal.Key = Dependent.ForeignKey"
        private static readonly Regex RelationshipPattern = new Regex(
            @"^\s*(\w+)\s+([\|\}o\{\-\>]+)\s+(\w+)\s*:\s*""([^""]+)""",
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
            var (relType, navType) = ParseRelationshipSymbol(relationshipSymbol);
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
                NavigationType = navType
            };
        }

        private static (RelationshipType?, NavigationType) ParseRelationshipSymbol(string symbol)
        {
            // Remove all spaces for easier matching
            var cleaned = symbol.Replace(" ", "");

            return cleaned switch
            {
                "||--o{" => (RelationshipType.OneToMany, NavigationType.Bidirectional),
                "||--||" => (RelationshipType.OneToOne, NavigationType.Bidirectional),
                "||-->o{" => (RelationshipType.OneToMany, NavigationType.Unidirectional),
                "||-->||" => (RelationshipType.OneToOne, NavigationType.Unidirectional),
                "o{--||" => (RelationshipType.ManyToOne, NavigationType.Bidirectional),
                "o{-->||" => (RelationshipType.ManyToOne, NavigationType.Unidirectional),
                _ => (null, NavigationType.Bidirectional)
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

