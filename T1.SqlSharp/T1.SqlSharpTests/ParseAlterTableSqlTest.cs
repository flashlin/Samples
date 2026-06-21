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

    [Test]
    public void Alter_table_enable_trigger_all()
    {
        var sql = "ALTER TABLE Users ENABLE TRIGGER ALL";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlAlterTableStatement
        {
            TableName = "Users",
            Action = new SqlAlterTableToggleTrigger { Enable = true, AllTriggers = true }
        });
    }

    [Test]
    public void Alter_table_disable_trigger_named()
    {
        var sql = "ALTER TABLE Users DISABLE TRIGGER trg_audit, trg_log";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlAlterTableStatement
        {
            TableName = "Users",
            Action = new SqlAlterTableToggleTrigger
            {
                Enable = false,
                TriggerNames = ["trg_audit", "trg_log"]
            }
        });
    }

    [Test]
    public void Alter_table_check_constraint_all()
    {
        var sql = "ALTER TABLE Users CHECK CONSTRAINT ALL";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlAlterTableStatement
        {
            TableName = "Users",
            Action = new SqlAlterTableCheckConstraint { Check = true, AllConstraints = true }
        });
    }

    [Test]
    public void Alter_table_nocheck_constraint_named()
    {
        var sql = "ALTER TABLE Users NOCHECK CONSTRAINT FK_Users_Orders";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlAlterTableStatement
        {
            TableName = "Users",
            Action = new SqlAlterTableCheckConstraint
            {
                Check = false,
                ConstraintNames = ["FK_Users_Orders"]
            }
        });
    }

    [Test]
    public void Alter_table_with_nocheck_add_constraint()
    {
        var sql = "ALTER TABLE Orders WITH NOCHECK ADD CONSTRAINT FK_Orders_Users FOREIGN KEY (UserId) REFERENCES Users (Id)";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlAlterTableStatement
        {
            TableName = "Orders",
            Action = new SqlAlterTableAddConstraint
            {
                WithCheck = false,
                Constraint = new SqlConstraintForeignKey
                {
                    ConstraintName = "FK_Orders_Users",
                    Columns = [new SqlConstraintColumn { ColumnName = "UserId", Order = "" }],
                    ReferencedTableName = "Users",
                    RefColumn = "Id"
                }
            }
        });
    }
}
