using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseAlterTableSqlTest
{
    [Test]
    public void Alter_table_add_single_column()
    {
        var sql = "ALTER TABLE Users ADD Age INT";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlAlterTableStatement
        {
            TableName = "Users",
            Action = new SqlAlterTableAddColumns
            {
                Columns =
                [
                    new SqlColumnDefinition { ColumnName = "Age", DataType = "INT" }
                ]
            }
        });
    }

    [Test]
    public void Alter_table_add_multiple_columns()
    {
        var sql = "ALTER TABLE Users ADD Age INT, Nickname VARCHAR(50) NULL";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlAlterTableStatement
        {
            TableName = "Users",
            Action = new SqlAlterTableAddColumns
            {
                Columns =
                [
                    new SqlColumnDefinition { ColumnName = "Age", DataType = "INT" },
                    new SqlColumnDefinition
                    {
                        ColumnName = "Nickname",
                        DataType = "VARCHAR",
                        DataSize = new SqlDataSize { Size = "50" },
                        IsNullable = true
                    }
                ]
            }
        });
    }

    [Test]
    public void Alter_table_drop_column()
    {
        var sql = "ALTER TABLE Users DROP COLUMN Age, Nickname";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlAlterTableStatement
        {
            TableName = "Users",
            Action = new SqlAlterTableDropColumn
            {
                ColumnNames = ["Age", "Nickname"]
            }
        });
    }

    [Test]
    public void Alter_table_alter_column()
    {
        var sql = "ALTER TABLE Users ALTER COLUMN Age BIGINT NOT NULL";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlAlterTableStatement
        {
            TableName = "Users",
            Action = new SqlAlterTableAlterColumn
            {
                Column = new SqlColumnDefinition
                {
                    ColumnName = "Age",
                    DataType = "BIGINT",
                    IsNullable = false
                }
            }
        });
    }

    [Test]
    public void Alter_table_add_constraint_primary_key()
    {
        var sql = "ALTER TABLE Users ADD CONSTRAINT PK_Users PRIMARY KEY (Id)";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlAlterTableStatement
        {
            TableName = "Users",
            Action = new SqlAlterTableAddConstraint
            {
                Constraint = new SqlConstraintPrimaryKeyOrUnique
                {
                    ConstraintName = "PK_Users",
                    ConstraintType = "PRIMARY KEY",
                    Clustered = "",
                    Columns =
                    [
                        new SqlConstraintColumn { ColumnName = "Id", Order = "" }
                    ]
                }
            }
        });
    }

    [Test]
    public void Alter_table_drop_constraint()
    {
        var sql = "ALTER TABLE Users DROP CONSTRAINT PK_Users";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlAlterTableStatement
        {
            TableName = "Users",
            Action = new SqlAlterTableDropConstraint
            {
                ConstraintNames = ["PK_Users"]
            }
        });
    }
}
