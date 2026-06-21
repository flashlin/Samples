using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseCreateIndexSqlTest
{
    [Test]
    public void Create_index_simple()
    {
        var sql = "CREATE INDEX ix_Name ON Customer (Name)";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateIndexStatement
        {
            IndexName = "ix_Name",
            TableName = "Customer",
            Columns =
            [
                new SqlConstraintColumn { ColumnName = "Name", Order = "" }
            ]
        });
    }

    [Test]
    public void Create_spatial_index()
    {
        var sql = "CREATE SPATIAL INDEX sidx ON Locations (GeoCol)";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateIndexStatement
        {
            IsSpatial = true,
            IndexName = "sidx",
            TableName = "Locations",
            Columns =
            [
                new SqlConstraintColumn { ColumnName = "GeoCol", Order = "" }
            ]
        });
    }

    [Test]
    public void Create_clustered_columnstore_index()
    {
        var sql = "CREATE CLUSTERED COLUMNSTORE INDEX cci ON Orders";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateIndexStatement
        {
            IsColumnstore = true,
            Clustered = "CLUSTERED",
            IndexName = "cci",
            TableName = "Orders"
        });
    }

    [Test]
    public void Create_nonclustered_columnstore_index_with_columns()
    {
        var sql = "CREATE NONCLUSTERED COLUMNSTORE INDEX ncci ON Orders (OrderId, Amount)";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateIndexStatement
        {
            IsColumnstore = true,
            Clustered = "NONCLUSTERED",
            IndexName = "ncci",
            TableName = "Orders",
            Columns =
            [
                new SqlConstraintColumn { ColumnName = "OrderId", Order = "" },
                new SqlConstraintColumn { ColumnName = "Amount", Order = "" }
            ]
        });
    }

    [Test]
    public void Create_unique_nonclustered_index_with_order()
    {
        var sql = "CREATE UNIQUE NONCLUSTERED INDEX ix ON Customer (Name ASC, Age DESC)";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateIndexStatement
        {
            IsUnique = true,
            Clustered = "NONCLUSTERED",
            IndexName = "ix",
            TableName = "Customer",
            Columns =
            [
                new SqlConstraintColumn { ColumnName = "Name", Order = "ASC" },
                new SqlConstraintColumn { ColumnName = "Age", Order = "DESC" }
            ]
        });
    }

    [Test]
    public void Create_clustered_index_with_include()
    {
        var sql = "CREATE CLUSTERED INDEX ix ON t (a) INCLUDE (b, c)";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateIndexStatement
        {
            Clustered = "CLUSTERED",
            IndexName = "ix",
            TableName = "t",
            Columns =
            [
                new SqlConstraintColumn { ColumnName = "a", Order = "" }
            ],
            IncludeColumns = ["b", "c"]
        });
    }

    [Test]
    public void Create_filtered_index_with_where()
    {
        var sql = "CREATE INDEX ix ON t (a) WHERE a > 0";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlCreateIndexStatement
        {
            IndexName = "ix",
            TableName = "t",
            Columns =
            [
                new SqlConstraintColumn { ColumnName = "a", Order = "" }
            ],
            Where = new SqlConditionExpression
            {
                Left = new SqlFieldExpr { FieldName = "a" },
                ComparisonOperator = ComparisonOperator.GreaterThan,
                Right = new SqlValue { SqlType = SqlType.IntValue, Value = "0" }
            }
        });
    }
}
