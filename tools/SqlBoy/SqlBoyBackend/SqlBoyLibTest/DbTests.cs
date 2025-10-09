using FluentAssertions;
using NUnit.Framework;
using SqlBoyLib;

namespace SqlBoyLibTest;

[TestFixture]
public class DbTests
{
    [Test]
    public void From_ShouldCreateQueryBuilder()
    {
        var builder = Db.From<Customer>("Employee");

        builder.Should().NotBeNull();
        builder.Should().BeOfType<SqlQueryBuilder<Customer>>();
    }

    [Test]
    public void Where_WithSimpleCondition_ShouldGenerateCorrectSql()
    {
        var query = Db.From<Customer>("Employee")
            .Where(x => x.Id > 1)
            .Build();

        query.Statement.Should().Be("SELECT * FROM Employee WHERE Id > @p1");
        query.ParameterDefinitions.Should().Be("@p1 INT");
        query.Parameters.Should().ContainKey("@p1");
        query.Parameters["@p1"].Should().Be(1);
    }

    [Test]
    public void Where_WithMultipleConditions_ShouldGenerateCorrectSql()
    {
        var query = Db.From<Customer>("Employee")
            .Where(x => x.Id > 1 && x.Name == "John")
            .Build();

        query.Statement.Should().Be("SELECT * FROM Employee WHERE (Id > @p1 AND Name = @p2)");
        query.ParameterDefinitions.Should().Be("@p1 INT, @p2 NVARCHAR(MAX)");
        query.Parameters.Should().ContainKey("@p1");
        query.Parameters.Should().ContainKey("@p2");
        query.Parameters["@p1"].Should().Be(1);
        query.Parameters["@p2"].Should().Be("John");
    }

    [Test]
    public void Where_WithComplexConditions_ShouldGenerateCorrectSql()
    {
        var query = Db.From<Customer>("Employee")
            .Where(x => (x.Age >= 18 && x.Age <= 65) || x.Department == "IT")
            .Build();

        query.Statement.Should().Be("SELECT * FROM Employee WHERE ((Age >= @p1 AND Age <= @p2) OR Department = @p3)");
        query.ParameterDefinitions.Should().Be("@p1 INT, @p2 INT, @p3 NVARCHAR(MAX)");
        query.Parameters["@p1"].Should().Be(18);
        query.Parameters["@p2"].Should().Be(65);
        query.Parameters["@p3"].Should().Be("IT");
    }

    [Test]
    public void Where_WithNotEqualOperator_ShouldGenerateCorrectSql()
    {
        var query = Db.From<Customer>("Employee")
            .Where(x => x.Name != "Admin")
            .Build();

        query.Statement.Should().Be("SELECT * FROM Employee WHERE Name <> @p1");
        query.Parameters["@p1"].Should().Be("Admin");
    }

    [Test]
    public void Where_WithLessThanOperator_ShouldGenerateCorrectSql()
    {
        var query = Db.From<Customer>("Employee")
            .Where(x => x.Age < 30)
            .Build();

        query.Statement.Should().Be("SELECT * FROM Employee WHERE Age < @p1");
        query.Parameters["@p1"].Should().Be(30);
    }

    [Test]
    public void Where_WithGreaterThanOrEqualOperator_ShouldGenerateCorrectSql()
    {
        var query = Db.From<Customer>("Employee")
            .Where(x => x.Salary >= 50000)
            .Build();

        query.Statement.Should().Be("SELECT * FROM Employee WHERE Salary >= @p1");
        query.Parameters["@p1"].Should().Be(50000);
    }

    [Test]
    public void OrderBy_ShouldGenerateCorrectSql()
    {
        var query = Db.From<Customer>("Employee")
            .Where(x => x.Id > 1)
            .OrderBy(x => x.Name)
            .Build();

        query.Statement.Should().Be("SELECT * FROM Employee WHERE Id > @p1 ORDER BY Name ASC");
        query.Parameters["@p1"].Should().Be(1);
    }

    [Test]
    public void OrderByDescending_ShouldGenerateCorrectSql()
    {
        var query = Db.From<Customer>("Employee")
            .Where(x => x.Age > 18)
            .OrderByDescending(x => x.Salary)
            .Build();

        query.Statement.Should().Be("SELECT * FROM Employee WHERE Age > @p1 ORDER BY Salary DESC");
        query.Parameters["@p1"].Should().Be(18);
    }

    [Test]
    public void OrderBy_WithMultipleColumns_ShouldGenerateCorrectSql()
    {
        var query = Db.From<Customer>("Employee")
            .Where(x => x.Age > 18)
            .OrderBy(x => x.Department)
            .OrderByDescending(x => x.Salary)
            .Build();

        query.Statement.Should().Be("SELECT * FROM Employee WHERE Age > @p1 ORDER BY Department ASC, Salary DESC");
        query.Parameters["@p1"].Should().Be(18);
    }

    [Test]
    public void Build_WithoutWhere_ShouldGenerateSelectAllSql()
    {
        var query = Db.From<Customer>("Employee")
            .Build();

        query.Statement.Should().Be("SELECT * FROM Employee");
        query.ParameterDefinitions.Should().BeEmpty();
        query.Parameters.Should().BeEmpty();
    }

    [Test]
    public void ToExecuteSql_ShouldGenerateSpExecuteSqlFormat()
    {
        var query = Db.From<Customer>("Employee")
            .Where(x => x.Id > 1 && x.Name == "John")
            .OrderBy(x => x.Age)
            .Build();

        var sql = query.ToExecuteSql();

        sql.Should().Contain("EXEC sys.sp_executesql");
        sql.Should().Contain("@stmt = N'SELECT * FROM Employee WHERE (Id > @p1 AND Name = @p2) ORDER BY Age ASC'");
        sql.Should().Contain("@params = N'@p1 INT, @p2 NVARCHAR(MAX)'");
        sql.Should().Contain("@p1 = 1");
        sql.Should().Contain("@p2 = N'John'");
    }

    [Test]
    public void ToExecuteSql_WithoutParameters_ShouldGenerateSimpleSpExecuteSql()
    {
        var query = Db.From<Customer>("Employee")
            .Build();

        var sql = query.ToExecuteSql();

        sql.Should().Be("EXEC sys.sp_executesql\n  @stmt = N'SELECT * FROM Employee'");
    }

    [Test]
    public void Where_WithOrCondition_ShouldGenerateCorrectSql()
    {
        var query = Db.From<Customer>("Employee")
            .Where(x => x.Department == "IT" || x.Department == "HR")
            .Build();

        query.Statement.Should().Be("SELECT * FROM Employee WHERE (Department = @p1 OR Department = @p2)");
        query.Parameters["@p1"].Should().Be("IT");
        query.Parameters["@p2"].Should().Be("HR");
    }

    [Test]
    public void CompleteQuery_WithAllFeatures_ShouldGenerateCorrectSql()
    {
        var query = Db.From<Customer>("Employee")
            .Where(x => x.Age > 25 && x.Salary >= 40000 && x.Department == "IT")
            .OrderBy(x => x.Name)
            .OrderByDescending(x => x.HireDate)
            .Build();

        query.Statement.Should().Be("SELECT * FROM Employee WHERE ((Age > @p1 AND Salary >= @p2) AND Department = @p3) ORDER BY Name ASC, HireDate DESC");
        query.Parameters["@p1"].Should().Be(25);
        query.Parameters["@p2"].Should().Be(40000);
        query.Parameters["@p3"].Should().Be("IT");

        var sql = query.ToExecuteSql();
        sql.Should().Contain("EXEC sys.sp_executesql");
        sql.Should().Contain("@p1 = 25");
        sql.Should().Contain("@p2 = 40000");
        sql.Should().Contain("@p3 = N'IT'");
    }
}

