using FluentAssertions;
using NUnit.Framework;
using T1.EfCodeFirstGenerateCli.ConfigParser;
using T1.EfCodeFirstGenerateCli.Models;

namespace T1.EfCodeFirstGenerateCliTest.Tests
{
    [TestFixture]
    public class MermaidRelationshipParserTests
    {
        [Test]
        public void ParseRelationship_OneToManyBidirectional_ParsesCorrectly()
        {
            var line = "User ||--o{ Order : \"User.Id = Order.UserId\"";
            
            var result = MermaidRelationshipParser.ParseRelationship(line);
            
            var expected = new EntityRelationship
            {
                PrincipalEntity = "User",
                PrincipalKey = "Id",
                DependentEntity = "Order",
                ForeignKey = "UserId",
                Type = RelationshipType.OneToMany,
                NavigationType = NavigationType.Bidirectional
            };
            
            result.Should().BeEquivalentTo(expected);
        }

        [Test]
        public void ParseRelationship_OneToOneBidirectional_ParsesCorrectly()
        {
            var line = "User ||--|| Profile : \"User.Id = Profile.UserId\"";
            
            var result = MermaidRelationshipParser.ParseRelationship(line);
            
            var expected = new EntityRelationship
            {
                PrincipalEntity = "User",
                PrincipalKey = "Id",
                DependentEntity = "Profile",
                ForeignKey = "UserId",
                Type = RelationshipType.OneToOne,
                NavigationType = NavigationType.Bidirectional
            };
            
            result.Should().BeEquivalentTo(expected);
        }

        [Test]
        public void ParseRelationship_OneToManyUnidirectional_ParsesCorrectly()
        {
            var line = "Category ||-->o{ Product : \"Category.Id = Product.CategoryId\"";
            
            var result = MermaidRelationshipParser.ParseRelationship(line);
            
            var expected = new EntityRelationship
            {
                PrincipalEntity = "Category",
                PrincipalKey = "Id",
                DependentEntity = "Product",
                ForeignKey = "CategoryId",
                Type = RelationshipType.OneToMany,
                NavigationType = NavigationType.Unidirectional
            };
            
            result.Should().BeEquivalentTo(expected);
        }

        [Test]
        public void ParseRelationship_OneToOneUnidirectional_ParsesCorrectly()
        {
            var line = "User ||-->|| Profile : \"User.Id = Profile.UserId\"";
            
            var result = MermaidRelationshipParser.ParseRelationship(line);
            
            var expected = new EntityRelationship
            {
                PrincipalEntity = "User",
                PrincipalKey = "Id",
                DependentEntity = "Profile",
                ForeignKey = "UserId",
                Type = RelationshipType.OneToOne,
                NavigationType = NavigationType.Unidirectional
            };
            
            result.Should().BeEquivalentTo(expected);
        }

        [Test]
        public void ParseRelationship_ManyToOneBidirectional_ParsesCorrectly()
        {
            var line = "Order o{--|| User : \"Order.UserId = User.Id\"";
            
            var result = MermaidRelationshipParser.ParseRelationship(line);
            
            var expected = new EntityRelationship
            {
                PrincipalEntity = "Order",
                PrincipalKey = "UserId",
                DependentEntity = "User",
                ForeignKey = "Id",
                Type = RelationshipType.ManyToOne,
                NavigationType = NavigationType.Bidirectional
            };
            
            result.Should().BeEquivalentTo(expected);
        }

        [Test]
        public void ParseRelationship_ManyToOneUnidirectional_ParsesCorrectly()
        {
            var line = "Order o{-->|| User : \"Order.UserId = User.Id\"";
            
            var result = MermaidRelationshipParser.ParseRelationship(line);
            
            var expected = new EntityRelationship
            {
                PrincipalEntity = "Order",
                PrincipalKey = "UserId",
                DependentEntity = "User",
                ForeignKey = "Id",
                Type = RelationshipType.ManyToOne,
                NavigationType = NavigationType.Unidirectional
            };
            
            result.Should().BeEquivalentTo(expected);
        }

        [Test]
        public void ParseRelationship_WithSpaces_ParsesCorrectly()
        {
            var line = "  User  ||--o{  Order  :  \"User.Id = Order.UserId\"  ";
            
            var result = MermaidRelationshipParser.ParseRelationship(line);
            
            var expected = new EntityRelationship
            {
                PrincipalEntity = "User",
                PrincipalKey = "Id",
                DependentEntity = "Order",
                ForeignKey = "UserId",
                Type = RelationshipType.OneToMany,
                NavigationType = NavigationType.Bidirectional
            };
            
            result.Should().BeEquivalentTo(expected);
        }

        [Test]
        public void ParseRelationship_InvalidSyntax_ReturnsNull()
        {
            var line = "InvalidLine";
            
            var result = MermaidRelationshipParser.ParseRelationship(line);
            
            result.Should().BeNull();
        }

        [Test]
        public void ParseRelationship_InvalidRelationshipSymbol_ReturnsNull()
        {
            var line = "User ?????? Order : \"User.Id = Order.UserId\"";
            
            var result = MermaidRelationshipParser.ParseRelationship(line);
            
            result.Should().BeNull();
        }

        [Test]
        public void ParseRelationship_InvalidKeyMapping_ReturnsNull()
        {
            var line = "User ||--o{ Order : \"InvalidMapping\"";
            
            var result = MermaidRelationshipParser.ParseRelationship(line);
            
            result.Should().BeNull();
        }

        [Test]
        public void ParseRelationship_MissingQuotes_ReturnsNull()
        {
            var line = "User ||--o{ Order : User.Id = Order.UserId";
            
            var result = MermaidRelationshipParser.ParseRelationship(line);
            
            result.Should().BeNull();
        }

        [Test]
        public void ParseRelationship_OneToZeroOrOneBidirectional_ParsesCorrectly()
        {
            var line = "User ||--o| Profile : \"User.Id = Profile.UserId\"";
            
            var result = MermaidRelationshipParser.ParseRelationship(line);
            
            var expected = new EntityRelationship
            {
                PrincipalEntity = "User",
                PrincipalKey = "Id",
                DependentEntity = "Profile",
                ForeignKey = "UserId",
                Type = RelationshipType.OneToOne,
                NavigationType = NavigationType.Bidirectional,
                IsPrincipalOptional = false,
                IsDependentOptional = true
            };
            
            result.Should().BeEquivalentTo(expected);
        }

        [Test]
        public void ParseRelationship_OneToZeroOrOneUnidirectional_ParsesCorrectly()
        {
            var line = "User ||-->o| Profile : \"User.Id = Profile.UserId\"";
            
            var result = MermaidRelationshipParser.ParseRelationship(line);
            
            var expected = new EntityRelationship
            {
                PrincipalEntity = "User",
                PrincipalKey = "Id",
                DependentEntity = "Profile",
                ForeignKey = "UserId",
                Type = RelationshipType.OneToOne,
                NavigationType = NavigationType.Unidirectional,
                IsPrincipalOptional = false,
                IsDependentOptional = true
            };
            
            result.Should().BeEquivalentTo(expected);
        }

        [Test]
        public void ParseRelationship_ZeroOrOneToOneBidirectional_ParsesCorrectly()
        {
            var line = "Profile o|--|| User : \"Profile.UserId = User.Id\"";
            
            var result = MermaidRelationshipParser.ParseRelationship(line);
            
            var expected = new EntityRelationship
            {
                PrincipalEntity = "Profile",
                PrincipalKey = "UserId",
                DependentEntity = "User",
                ForeignKey = "Id",
                Type = RelationshipType.OneToOne,
                NavigationType = NavigationType.Bidirectional,
                IsPrincipalOptional = true,
                IsDependentOptional = false
            };
            
            result.Should().BeEquivalentTo(expected);
        }

        [Test]
        public void ParseRelationship_ZeroOrOneToOneUnidirectional_ParsesCorrectly()
        {
            var line = "Profile o|-->|| User : \"Profile.UserId = User.Id\"";
            
            var result = MermaidRelationshipParser.ParseRelationship(line);
            
            var expected = new EntityRelationship
            {
                PrincipalEntity = "Profile",
                PrincipalKey = "UserId",
                DependentEntity = "User",
                ForeignKey = "Id",
                Type = RelationshipType.OneToOne,
                NavigationType = NavigationType.Unidirectional,
                IsPrincipalOptional = true,
                IsDependentOptional = false
            };
            
            result.Should().BeEquivalentTo(expected);
        }
    }
}

