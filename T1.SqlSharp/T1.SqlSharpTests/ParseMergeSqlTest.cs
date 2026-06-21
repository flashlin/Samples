using T1.SqlSharp.Expressions;
using T1.SqlSharp.Extensions;

namespace T1.SqlSharpTests;

[TestFixture]
public class ParseMergeSqlTest
{
    [Test]
    public void Merge_matched_update()
    {
        var sql = "MERGE INTO Target AS t USING Source AS s ON t.id = s.id "
                  + "WHEN MATCHED THEN UPDATE SET t.name = s.name";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlMergeStatement
        {
            Target = new SqlTableSource { TableName = "Target", Alias = "t" },
            Source = new SqlTableSource { TableName = "Source", Alias = "s" },
            OnCondition = new SqlConditionExpression
            {
                Left = new SqlFieldExpr { FieldName = "t.id" },
                ComparisonOperator = ComparisonOperator.Equal,
                Right = new SqlFieldExpr { FieldName = "s.id" }
            },
            WhenClauses =
            [
                new SqlMergeWhenClause
                {
                    MatchType = MergeMatchType.Matched,
                    Action = new SqlMergeUpdateAction
                    {
                        SetClauses =
                        [
                            new SqlAssignExpr
                            {
                                Left = new SqlFieldExpr { FieldName = "t.name" },
                                Right = new SqlFieldExpr { FieldName = "s.name" }
                            }
                        ]
                    }
                }
            ]
        });
    }

    [Test]
    public void Merge_full_three_when_clauses()
    {
        var sql = "MERGE INTO Target AS t USING Source AS s ON t.id = s.id "
                  + "WHEN MATCHED THEN UPDATE SET t.name = s.name "
                  + "WHEN NOT MATCHED THEN INSERT (id, name) VALUES (s.id, s.name) "
                  + "WHEN NOT MATCHED BY SOURCE THEN DELETE;";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlMergeStatement
        {
            Target = new SqlTableSource { TableName = "Target", Alias = "t" },
            Source = new SqlTableSource { TableName = "Source", Alias = "s" },
            OnCondition = new SqlConditionExpression
            {
                Left = new SqlFieldExpr { FieldName = "t.id" },
                ComparisonOperator = ComparisonOperator.Equal,
                Right = new SqlFieldExpr { FieldName = "s.id" }
            },
            WhenClauses =
            [
                new SqlMergeWhenClause
                {
                    MatchType = MergeMatchType.Matched,
                    Action = new SqlMergeUpdateAction
                    {
                        SetClauses =
                        [
                            new SqlAssignExpr
                            {
                                Left = new SqlFieldExpr { FieldName = "t.name" },
                                Right = new SqlFieldExpr { FieldName = "s.name" }
                            }
                        ]
                    }
                },
                new SqlMergeWhenClause
                {
                    MatchType = MergeMatchType.NotMatchedByTarget,
                    Action = new SqlMergeInsertAction
                    {
                        Columns = ["id", "name"],
                        Values =
                        [
                            new SqlFieldExpr { FieldName = "s.id" },
                            new SqlFieldExpr { FieldName = "s.name" }
                        ]
                    }
                },
                new SqlMergeWhenClause
                {
                    MatchType = MergeMatchType.NotMatchedBySource,
                    Action = new SqlMergeDeleteAction()
                }
            ]
        });
    }

    [Test]
    public void Merge_without_aliases()
    {
        var sql = "MERGE Target USING Source ON Target.id = Source.id WHEN MATCHED THEN DELETE";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlMergeStatement
        {
            Target = new SqlTableSource { TableName = "Target" },
            Source = new SqlTableSource { TableName = "Source" },
            OnCondition = new SqlConditionExpression
            {
                Left = new SqlFieldExpr { FieldName = "Target.id" },
                ComparisonOperator = ComparisonOperator.Equal,
                Right = new SqlFieldExpr { FieldName = "Source.id" }
            },
            WhenClauses =
            [
                new SqlMergeWhenClause
                {
                    MatchType = MergeMatchType.Matched,
                    Action = new SqlMergeDeleteAction()
                }
            ]
        });
    }

    [Test]
    public void Merge_not_matched_insert_default_values()
    {
        var sql = "MERGE INTO Target AS t USING Source AS s ON t.id = s.id "
                  + "WHEN NOT MATCHED THEN INSERT DEFAULT VALUES";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlMergeStatement
        {
            Target = new SqlTableSource { TableName = "Target", Alias = "t" },
            Source = new SqlTableSource { TableName = "Source", Alias = "s" },
            OnCondition = new SqlConditionExpression
            {
                Left = new SqlFieldExpr { FieldName = "t.id" },
                ComparisonOperator = ComparisonOperator.Equal,
                Right = new SqlFieldExpr { FieldName = "s.id" }
            },
            WhenClauses =
            [
                new SqlMergeWhenClause
                {
                    MatchType = MergeMatchType.NotMatchedByTarget,
                    Action = new SqlMergeInsertAction { IsDefaultValues = true }
                }
            ]
        });
    }

    [Test]
    public void Merge_top_n()
    {
        var sql = "MERGE TOP (10) INTO Target AS t USING Source AS s ON t.id = s.id "
                  + "WHEN MATCHED THEN DELETE";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlMergeStatement
        {
            Top = new SqlTopClause
            {
                Expression = new SqlParenthesizedExpression
                {
                    Inner = new SqlValue { SqlType = SqlType.IntValue, Value = "10" }
                }
            },
            Target = new SqlTableSource { TableName = "Target", Alias = "t" },
            Source = new SqlTableSource { TableName = "Source", Alias = "s" },
            OnCondition = new SqlConditionExpression
            {
                Left = new SqlFieldExpr { FieldName = "t.id" },
                ComparisonOperator = ComparisonOperator.Equal,
                Right = new SqlFieldExpr { FieldName = "s.id" }
            },
            WhenClauses =
            [
                new SqlMergeWhenClause { MatchType = MergeMatchType.Matched, Action = new SqlMergeDeleteAction() }
            ]
        });
    }

    [Test]
    public void Merge_target_table_hint()
    {
        var sql = "MERGE INTO Target WITH (HOLDLOCK) USING Source ON Target.id = Source.id "
                  + "WHEN MATCHED THEN DELETE";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlMergeStatement
        {
            Target = new SqlTableSource { TableName = "Target", Withs = [new SqlHint { Name = "HOLDLOCK" }] },
            Source = new SqlTableSource { TableName = "Source" },
            OnCondition = new SqlConditionExpression
            {
                Left = new SqlFieldExpr { FieldName = "Target.id" },
                ComparisonOperator = ComparisonOperator.Equal,
                Right = new SqlFieldExpr { FieldName = "Source.id" }
            },
            WhenClauses =
            [
                new SqlMergeWhenClause { MatchType = MergeMatchType.Matched, Action = new SqlMergeDeleteAction() }
            ]
        });
    }

    [Test]
    public void Merge_with_output_and_option()
    {
        var sql = "MERGE INTO Target AS t USING Source AS s ON t.id = s.id "
                  + "WHEN MATCHED THEN DELETE "
                  + "OUTPUT deleted.id "
                  + "OPTION (RECOMPILE)";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlMergeStatement
        {
            Target = new SqlTableSource { TableName = "Target", Alias = "t" },
            Source = new SqlTableSource { TableName = "Source", Alias = "s" },
            OnCondition = new SqlConditionExpression
            {
                Left = new SqlFieldExpr { FieldName = "t.id" },
                ComparisonOperator = ComparisonOperator.Equal,
                Right = new SqlFieldExpr { FieldName = "s.id" }
            },
            WhenClauses =
            [
                new SqlMergeWhenClause { MatchType = MergeMatchType.Matched, Action = new SqlMergeDeleteAction() }
            ],
            Output = new SqlOutputClause
            {
                Columns = [new SelectColumn { Field = new SqlFieldExpr { FieldName = "deleted.id" } }]
            },
            Option = new SqlOptionClause
            {
                Hints = [new SqlQueryHint { Name = "RECOMPILE" }]
            }
        });
    }

    [Test]
    public void Merge_when_matched_and_condition_then_delete()
    {
        var sql = "MERGE INTO Target AS t USING Source AS s ON t.id = s.id "
                  + "WHEN MATCHED AND s.active = 0 THEN DELETE";
        var rc = sql.ParseSql();
        rc.ShouldBe(new SqlMergeStatement
        {
            Target = new SqlTableSource { TableName = "Target", Alias = "t" },
            Source = new SqlTableSource { TableName = "Source", Alias = "s" },
            OnCondition = new SqlConditionExpression
            {
                Left = new SqlFieldExpr { FieldName = "t.id" },
                ComparisonOperator = ComparisonOperator.Equal,
                Right = new SqlFieldExpr { FieldName = "s.id" }
            },
            WhenClauses =
            [
                new SqlMergeWhenClause
                {
                    MatchType = MergeMatchType.Matched,
                    AndCondition = new SqlConditionExpression
                    {
                        Left = new SqlFieldExpr { FieldName = "s.active" },
                        ComparisonOperator = ComparisonOperator.Equal,
                        Right = new SqlValue { SqlType = SqlType.IntValue, Value = "0" }
                    },
                    Action = new SqlMergeDeleteAction()
                }
            ]
        });
    }
}
