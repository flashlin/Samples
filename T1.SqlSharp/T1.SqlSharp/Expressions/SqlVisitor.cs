namespace T1.SqlSharp.Expressions;

public class SqlVisitor
{
    private readonly List<SqlExpressionNode> _expressions = new();
    private int _depth = 0;

    public List<SqlExpressionNode> Visit(ISqlExpression expression)
    {
        _expressions.Clear();
        AddSqlExpression(expression);
        expression.Accept(this);
        return _expressions;
    }

    private void AddSqlExpression(ISqlExpression expression)
    {
        _expressions.Add(new SqlExpressionNode()
        {
            Expression = expression,
            Depth = _depth
        });
    }

    public virtual void Visit_SqlArithmeticBinaryExpr(SqlArithmeticBinaryExpr expr)
    {
        AddSqlExpression(expr);
        _depth++;
        expr.Left.Accept(this);
        expr.Right.Accept(this);
        _depth--;
    }

    public virtual void Visit_AliasExpr(SqlAliasExpr expr)
    {
    }

    public virtual void Visit_AsExpr(SqlAsExpr expr)
    {
        AddSqlExpression(expr);
        expr.Instance.Accept(this);
        expr.As.Accept(this);
    }

    public virtual void Visit_ArithmeticBinaryExpr(SqlArithmeticBinaryExpr expr)
    {
        AddSqlExpression(expr);
        _depth++;
        expr.Left.Accept(this);
        expr.Right.Accept(this);
        _depth--;
    }

    public virtual void Visit_LogicalOperator(SqlLogicalOperator expr)
    {
    }

    public virtual void Visit_SelectColumn(SelectColumn expr)
    {
        AddSqlExpression(expr);
        expr.Field.Accept(this);
    }

    public virtual void Visit_SelectStatement(SelectStatement expr)
    {
        AddSqlExpression(expr);
        expr.Top?.Accept(this);
        expr.Columns.ForEach(x =>
        {
            x.Accept(this);
        });
        expr.FromSources.ForEach(x=>
        {
            x.Accept(this);
        });
        if(expr.ForXml != null)
        {
            expr.ForXml.Accept(this);
        }
        if(expr.ForJson != null)
        {
            expr.ForJson.Accept(this);
        }
        if(expr.Where != null)
        {
            expr.Where.Accept(this);
        }
        if(expr.OrderBy != null)
        {
            expr.OrderBy.Accept(this);
        }
        expr.Unions.ForEach(x=>
        {
            x.Accept(this);
        });
        if(expr.GroupBy != null)
        {
            expr.GroupBy.Accept(this);
        }
        if(expr.Having != null)
        {
            expr.Having.Accept(this);
        }
        if(expr.Window != null)
        {
            expr.Window.Accept(this);
        }
        if(expr.Option != null)
        {
            expr.Option.Accept(this);
        }
    }

    public virtual void Visit_WindowClause(SqlWindowClause expr)
    {
        AddSqlExpression(expr);
        expr.Definitions.ForEach(x => x.Accept(this));
    }

    public virtual void Visit_WindowDefinition(SqlWindowDefinition expr)
    {
        AddSqlExpression(expr);
        expr.PartitionBy.ForEach(x => x.Accept(this));
        expr.OrderColumns.ForEach(x => x.Accept(this));
        expr.Frame?.Accept(this);
    }

    public virtual void Visit_OverWindowName(SqlOverWindowName expr)
    {
        AddSqlExpression(expr);
        expr.Field.Accept(this);
    }

    public virtual void Visit_OptionClause(SqlOptionClause expr)
    {
        AddSqlExpression(expr);
        expr.Hints.ForEach(x=>x.Accept(this));
    }

    public virtual void Visit_QueryHint(SqlQueryHint expr)
    {
        AddSqlExpression(expr);
        expr.Arguments.ForEach(x=>x.Accept(this));
    }

    public virtual void Visit_SqlToken(SqlToken expr)
    {
    }

    public virtual void Visit_SqlValue(SqlValue expr)
    {
    }

    public virtual void Visit_AssignExpr(SqlAssignExpr expr)
    {
        AddSqlExpression(expr);
        expr.Left.Accept(this);
        expr.Right.Accept(this);
    }

    public virtual void Visit_BetweenValue(SqlBetweenValue expr)
    {
        AddSqlExpression(expr);
        expr.Start.Accept(this);
        expr.End.Accept(this);
    }

    public virtual void Visit_CaseClause(SqlCaseClause expr)
    {
        AddSqlExpression(expr);
        expr.Case?.Accept(this);
        expr.WhenThens.ForEach(x=>x.Accept(this));
        expr.Else?.Accept(this);
    }

    public virtual void Visit_WhenThen(SqlWhenThenClause expr)
    {
        AddSqlExpression(expr);
        expr.When.Accept(this);
        expr.Then.Accept(this);
    }

    public virtual void Visit_ColumnDefinition(SqlColumnDefinition expr)
    {
        AddSqlExpression(expr);
        expr.Constraints.ForEach(x=>x.Accept(this));
    }

    public virtual void Visit_ComparisonOperator(SqlComparisonOperator expr)
    {
    }

    public virtual void Visit_ComputedColumnDefinition(SqlComputedColumnDefinition expr)
    {
    }

    public virtual void Visit_ConditionExpression(SqlConditionExpression expr)
    {
        AddSqlExpression(expr);
        expr.Left.Accept(this);
        expr.Right.Accept(this);
    }

    public virtual void Visit_ConstraintColumn(SqlConstraintColumn expr)
    {
    }

    public virtual void Visit_ConstraintCheck(SqlConstraintCheck expr)
    {
        AddSqlExpression(expr);
        expr.Predicate.Accept(this);
    }

    public virtual void Visit_CollateExpression(SqlCollateExpression expr)
    {
        AddSqlExpression(expr);
        expr.Expression.Accept(this);
    }

    public virtual void Visit_ConstraintDefaultValue(SqlConstraintDefaultValue expr)
    {
    }

    public virtual void Visit_ConstraintForeignKey(SqlConstraintForeignKey expr)
    {
        AddSqlExpression(expr);
        expr.Columns.ForEach(x=>x.Accept(this));
    }

    public virtual void Visit_ConstraintPrimaryKeyOrUnique(SqlConstraintPrimaryKeyOrUnique expr)
    {
        AddSqlExpression(expr);
        expr.Columns.ForEach(x=>x.Accept(this));
        expr.WithToggles.ForEach(x=>x.Accept(this));
    }

    public virtual void Visit_CreateTableExpression(SqlCreateTableExpression expr)
    {
        AddSqlExpression(expr);
        expr.Columns.ForEach(x=>x.Accept(this));
        expr.Constraints.ForEach(x=>x.Accept(this));
    }

    public virtual void Visit_DataSize(SqlDataSize expr)
    {
    }

    public virtual void Visit_DataTypeWithSize(SqlDataTypeWithSize expr)
    {
    }

    public virtual void Visit_Distinct(SqlDistinct expr)
    {
    }

    public virtual void Visit_ExistsExpression(SqlExistsExpression expr)
    {
        AddSqlExpression(expr);
        expr.Query.Accept(this);
    }

    public virtual void Visit_FieldExpr(SqlFieldExpr expr)
    {
    }

    public virtual void Visit_ForXmlAutoClause(SqlForXmlAutoClause expr)
    {
        AddSqlExpression(expr);
        expr.CommonDirectives.ForEach(x=>x.Accept(this));
    }

    public virtual void Visit_ForXmlPathClause(SqlForXmlPathClause expr)
    {
        AddSqlExpression(expr);
        expr.CommonDirectives.ForEach(x=>x.Accept(this));
    }
    public virtual void Visit_ForXmlModeClause(SqlForXmlModeClause expr)
    {
        AddSqlExpression(expr);
        expr.CommonDirectives.ForEach(x=>x.Accept(this));
    }

    public virtual void Visit_ForXmlRootDirective(SqlForXmlRootDirective expr)
    {
        AddSqlExpression(expr);
        expr.RootName?.Accept(this);
    }

    public virtual void Visit_ForJsonClause(SqlForJsonClause expr)
    {
        AddSqlExpression(expr);
        expr.RootName?.Accept(this);
    }

    public virtual void Visit_FunctionExpression(SqlFunctionExpression expr)
    {
        AddSqlExpression(expr);
        expr.Parameters.ForEach(x=>x.Accept(this));
        expr.WithinGroup?.Accept(this);
    }

    public virtual void Visit_WithinGroupClause(SqlWithinGroupClause expr)
    {
        AddSqlExpression(expr);
        expr.Columns.ForEach(x=>x.Accept(this));
    }

    public virtual void Visit_ParenthesizedExpression(SqlParenthesizedExpression expr)
    {
        AddSqlExpression(expr);
        expr.Inner.Accept(this);
    }

    public virtual void Visit_GroupByClause(SqlGroupByClause expr)
    {
        AddSqlExpression(expr);
        expr.Columns.ForEach(x=>x.Accept(this));
        expr.GroupingSets.ForEach(x=>x.Accept(this));
    }

    public virtual void Visit_GroupingSet(SqlGroupingSet expr)
    {
        AddSqlExpression(expr);
        expr.Columns.ForEach(x=>x.Accept(this));
    }

    public virtual void Visit_HavingClause(SqlHavingClause expr)
    {
        AddSqlExpression(expr);
        expr.Condition.Accept(this);
    }

    public virtual void Visit_Hint(SqlHint expr)
    {
    }

    public virtual void Visit_Identity(SqlIdentity expr)
    {
    }

    public virtual void Visit_JoinTableCondition(SqlJoinTableCondition expr)
    {
        AddSqlExpression(expr);
        expr.JoinedTable.Accept(this);
        expr.OnCondition?.Accept(this);
    }

    public virtual void Visit_CommonTableExpression(SqlCommonTableExpression expr)
    {
        AddSqlExpression(expr);
        expr.Query.Accept(this);
    }

    public virtual void Visit_WithCte(SqlWithCte expr)
    {
        AddSqlExpression(expr);
        expr.CommonTableExpressions.ForEach(x => x.Accept(this));
        expr.Statement.Accept(this);
    }

    public virtual void Visit_NegativeValue(SqlNegativeValue expr)
    {
        AddSqlExpression(expr);
        expr.Value.Accept(this);
    }

    public virtual void Visit_NotExpression(SqlNotExpression expr)
    {
        AddSqlExpression(expr);
        expr.Value.Accept(this);
    }

    public virtual void Visit_NullValue(SqlNullValue expr)
    {
    }

    public virtual void Visit_DefaultValue(SqlDefaultValue expr)
    {
    }

    public virtual void Visit_OrderByClause(SqlOrderByClause expr)
    {
        AddSqlExpression(expr);
        expr.Columns.ForEach(x=>x.Accept(this));
    }

    public virtual void Visit_OrderColumn(SqlOrderColumn expr)
    {
        AddSqlExpression(expr);
        expr.ColumnName.Accept(this);
    }

    public virtual void Visit_OverOrderByClause(SqlOverOrderByClause expr)
    {
        AddSqlExpression(expr);
        expr.Field.Accept(this);
        expr.Columns.ForEach(x=>x.Accept(this));
        expr.Frame?.Accept(this);
    }

    public virtual void Visit_OverPartitionByClause(SqlOverPartitionByClause expr)
    {
        AddSqlExpression(expr);
        expr.Field.Accept(this);
        expr.By.ForEach(x=>x.Accept(this));
        expr.Columns.ForEach(x=>x.Accept(this));
        expr.Frame?.Accept(this);
    }

    public virtual void Visit_WindowFrameClause(SqlWindowFrameClause expr)
    {
        AddSqlExpression(expr);
        expr.Start.Accept(this);
        expr.End?.Accept(this);
    }

    public virtual void Visit_WindowFrameBound(SqlWindowFrameBound expr)
    {
        AddSqlExpression(expr);
        expr.Offset?.Accept(this);
    }

    public virtual void Visit_ParameterValue(SqlParameterValue expr)
    {
    }

    public virtual void Visit_PartitionBy(SqlPartitionBy expr)
    {
        AddSqlExpression(expr);
        expr.Columns.ForEach(x=>x.Accept(this));
    }

    public virtual void Visit_RankClause(SqlRankClause expr)
    {
        AddSqlExpression(expr);
        expr.PartitionBy?.Accept(this);
        expr.OrderBy.Accept(this);
    }

    public virtual void Visit_SearchCondition(SqlSearchCondition expr)
    {
        AddSqlExpression(expr);
        expr.Left.Accept(this);
        expr.Right?.Accept(this);
    }

    public virtual void Visit_SetValueStatement(SqlSetValueStatement expr)
    {
        AddSqlExpression(expr);
        expr.Name.Accept(this);
        expr.Value.Accept(this);
    }

    public virtual void Visit_AddExtendedProperty(SqlSpAddExtendedPropertyExpression expr)
    {
    }

    public virtual void Visit_TableHintIndex(SqlTableHintIndex expr)
    {
        AddSqlExpression(expr);
        expr.IndexValues.ForEach(x=>x.Accept(this));
    }

    public virtual void Visit_TableSource(SqlTableSource expr)
    {
        AddSqlExpression(expr);
        expr.TableSample?.Accept(this);
        expr.Withs.ForEach(x=>x.Accept(this));
    }

    public virtual void Visit_TableSampleClause(SqlTableSampleClause expr)
    {
        AddSqlExpression(expr);
        expr.SampleNumber.Accept(this);
        expr.RepeatableSeed?.Accept(this);
    }

    public virtual void Visit_Toggle(SqlToggle expr)
    {
    }

    public virtual void Visit_TopClause(SqlTopClause expr)
    {
        AddSqlExpression(expr);
        expr.Expression.Accept(this);
    }

    public virtual void Visit_UnaryExpr(SqlUnaryExpr expr)
    {
        AddSqlExpression(expr);
        expr.Operand.Accept(this);
    }

    public virtual void Visit_UnionSelect(SqlUnionSelect expr)
    {
        AddSqlExpression(expr);
        expr.SelectStatement.Accept(this);
    }

    public virtual void Visit_UnpivotClause(SqlUnpivotClause expr)
    {
        AddSqlExpression(expr);
        expr.NewColumn.Accept(this);
        expr.ForSource.Accept(this);
        expr.InColumns.ForEach(x=>x.Accept(this));
    }

    public virtual void Visit_PivotClause(SqlPivotClause expr)
    {
        AddSqlExpression(expr);
        expr.NewColumn.Accept(this);
        expr.ForSource.Accept(this);
        expr.InColumns.ForEach(x=>x.Accept(this));
    }

    public virtual void Visit_Values(SqlValues expr)
    {
        AddSqlExpression(expr);
        expr.Items.ForEach(x=>x.Accept(this));
    }

    public virtual void Visit_ChangeTableChanges(SqlChangeTableChanges expr)
    {
    }

    public virtual void Visit_ChangeTableVersion(SqlChangeTableVersion expr)
    {
    }

    public virtual void Visit_InsertStatement(SqlInsertStatement expr)
    {
        AddSqlExpression(expr);
        expr.Output?.Accept(this);
        expr.ValuesRows.ForEach(row => row.ForEach(value => value.Accept(this)));
        expr.SourceSelect?.Accept(this);
        expr.ExecSource?.Accept(this);
    }

    public virtual void Visit_OutputClause(SqlOutputClause expr)
    {
        AddSqlExpression(expr);
        expr.Columns.ForEach(column => column.Accept(this));
    }

    public virtual void Visit_UpdateStatement(SqlUpdateStatement expr)
    {
        AddSqlExpression(expr);
        expr.SetClauses.ForEach(clause => clause.Accept(this));
        expr.Output?.Accept(this);
        expr.FromSources.ForEach(source => source.Accept(this));
        expr.Where?.Accept(this);
    }

    public virtual void Visit_DeleteStatement(SqlDeleteStatement expr)
    {
        AddSqlExpression(expr);
        expr.Output?.Accept(this);
        expr.FromSources.ForEach(source => source.Accept(this));
        expr.Where?.Accept(this);
    }

    public virtual void Visit_MergeStatement(SqlMergeStatement expr)
    {
        AddSqlExpression(expr);
        expr.Target.Accept(this);
        expr.Source.Accept(this);
        expr.OnCondition.Accept(this);
        expr.WhenClauses.ForEach(clause => clause.Accept(this));
        expr.Output?.Accept(this);
    }

    public virtual void Visit_MergeWhenClause(SqlMergeWhenClause expr)
    {
        AddSqlExpression(expr);
        expr.AndCondition?.Accept(this);
        expr.Action.Accept(this);
    }

    public virtual void Visit_MergeUpdateAction(SqlMergeUpdateAction expr)
    {
        AddSqlExpression(expr);
        expr.SetClauses.ForEach(clause => clause.Accept(this));
    }

    public virtual void Visit_MergeDeleteAction(SqlMergeDeleteAction expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_MergeInsertAction(SqlMergeInsertAction expr)
    {
        AddSqlExpression(expr);
        expr.Values.ForEach(value => value.Accept(this));
    }

    public virtual void Visit_TruncateTableStatement(SqlTruncateTableStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_DropStatement(SqlDropStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_AlterTableStatement(SqlAlterTableStatement expr)
    {
        AddSqlExpression(expr);
        expr.Action.Accept(this);
    }

    public virtual void Visit_AlterTableAddColumns(SqlAlterTableAddColumns expr)
    {
        AddSqlExpression(expr);
        expr.Columns.ForEach(column => column.Accept(this));
    }

    public virtual void Visit_AlterTableAddConstraint(SqlAlterTableAddConstraint expr)
    {
        AddSqlExpression(expr);
        expr.Constraint.Accept(this);
    }

    public virtual void Visit_AlterTableDropColumn(SqlAlterTableDropColumn expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_AlterTableDropConstraint(SqlAlterTableDropConstraint expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_AlterTableAlterColumn(SqlAlterTableAlterColumn expr)
    {
        AddSqlExpression(expr);
        expr.Column.Accept(this);
    }

    public virtual void Visit_AlterTableToggleTrigger(SqlAlterTableToggleTrigger expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_AlterTableCheckConstraint(SqlAlterTableCheckConstraint expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_AlterTableAddElements(SqlAlterTableAddElements expr)
    {
        AddSqlExpression(expr);
        expr.Columns.ForEach(column => column.Accept(this));
        expr.Constraints.ForEach(constraint => constraint.Accept(this));
    }

    public virtual void Visit_CreateViewStatement(SqlCreateViewStatement expr)
    {
        AddSqlExpression(expr);
        expr.Query.Accept(this);
    }

    public virtual void Visit_CreateIndexStatement(SqlCreateIndexStatement expr)
    {
        AddSqlExpression(expr);
        expr.Columns.ForEach(column => column.Accept(this));
        expr.Where?.Accept(this);
    }

    public virtual void Visit_ExecStatement(SqlExecStatement expr)
    {
        AddSqlExpression(expr);
        expr.Arguments.ForEach(argument => argument.Accept(this));
        expr.DynamicSql?.Accept(this);
    }

    public virtual void Visit_ExecArgument(SqlExecArgument expr)
    {
        AddSqlExpression(expr);
        expr.Value.Accept(this);
    }

    public virtual void Visit_DeclareStatement(SqlDeclareStatement expr)
    {
        AddSqlExpression(expr);
        expr.Declarations.ForEach(declaration =>
        {
            declaration.InitialValue?.Accept(this);
            declaration.TableColumns.ForEach(column => column.Accept(this));
            declaration.CursorSource?.Accept(this);
        });
    }

    public virtual void Visit_BlockStatement(SqlBlockStatement expr)
    {
        AddSqlExpression(expr);
        expr.Statements.ForEach(statement => statement.Accept(this));
    }

    public virtual void Visit_IfStatement(SqlIfStatement expr)
    {
        AddSqlExpression(expr);
        expr.Condition.Accept(this);
        expr.Then.Accept(this);
        expr.Else?.Accept(this);
    }

    public virtual void Visit_WhileStatement(SqlWhileStatement expr)
    {
        AddSqlExpression(expr);
        expr.Condition.Accept(this);
        expr.Body.Accept(this);
    }

    public virtual void Visit_CreateProcedureStatement(SqlCreateProcedureStatement expr)
    {
        AddSqlExpression(expr);
        expr.Parameters.ForEach(parameter => parameter.DefaultValue?.Accept(this));
        expr.Body.Accept(this);
    }

    public virtual void Visit_ReturnStatement(SqlReturnStatement expr)
    {
        AddSqlExpression(expr);
        expr.Value?.Accept(this);
    }

    public virtual void Visit_CreateFunctionStatement(SqlCreateFunctionStatement expr)
    {
        AddSqlExpression(expr);
        expr.Parameters.ForEach(parameter => parameter.DefaultValue?.Accept(this));
        expr.ReturnTableColumns.ForEach(column => column.Accept(this));
        expr.Body.Accept(this);
    }

    public virtual void Visit_TryCatchStatement(SqlTryCatchStatement expr)
    {
        AddSqlExpression(expr);
        expr.TryStatements.ForEach(statement => statement.Accept(this));
        expr.CatchStatements.ForEach(statement => statement.Accept(this));
    }

    public virtual void Visit_TransactionStatement(SqlTransactionStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_PrintStatement(SqlPrintStatement expr)
    {
        AddSqlExpression(expr);
        expr.Value.Accept(this);
    }

    public virtual void Visit_ThrowStatement(SqlThrowStatement expr)
    {
        AddSqlExpression(expr);
        expr.ErrorNumber?.Accept(this);
        expr.Message?.Accept(this);
        expr.State?.Accept(this);
    }

    public virtual void Visit_RaiseErrorStatement(SqlRaiseErrorStatement expr)
    {
        AddSqlExpression(expr);
        expr.Message.Accept(this);
        expr.Severity.Accept(this);
        expr.State.Accept(this);
        expr.Arguments.ForEach(argument => argument.Accept(this));
    }

    public virtual void Visit_LoopControlStatement(SqlLoopControlStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_CreateTriggerStatement(SqlCreateTriggerStatement expr)
    {
        AddSqlExpression(expr);
        expr.Body.Accept(this);
    }

    public virtual void Visit_UseStatement(SqlUseStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_GoStatement(SqlGoStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_PermissionStatement(SqlPermissionStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_CreateSchemaStatement(SqlCreateSchemaStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_CreateDatabaseStatement(SqlCreateDatabaseStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_CursorOperationStatement(SqlCursorOperationStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_FetchStatement(SqlFetchStatement expr)
    {
        AddSqlExpression(expr);
        expr.RowCount?.Accept(this);
    }

    public virtual void Visit_SetOptionStatement(SqlSetOptionStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_WaitForStatement(SqlWaitForStatement expr)
    {
        AddSqlExpression(expr);
        expr.Time.Accept(this);
    }

    public virtual void Visit_GotoStatement(SqlGotoStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_LabelStatement(SqlLabelStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_KeywordStatement(SqlKeywordStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_CreateSequenceStatement(SqlCreateSequenceStatement expr)
    {
        AddSqlExpression(expr);
        expr.StartWith?.Accept(this);
        expr.IncrementBy?.Accept(this);
    }

    public virtual void Visit_NextValueForExpr(SqlNextValueForExpr expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_CreateTypeStatement(SqlCreateTypeStatement expr)
    {
        AddSqlExpression(expr);
        expr.TableColumns.ForEach(column => column.Accept(this));
    }

    public virtual void Visit_CreateSynonymStatement(SqlCreateSynonymStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_AlterIndexStatement(SqlAlterIndexStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_AtTimeZoneExpr(SqlAtTimeZoneExpr expr)
    {
        AddSqlExpression(expr);
        expr.Expression.Accept(this);
        expr.TimeZone.Accept(this);
    }

    public virtual void Visit_AlterDatabaseStatement(SqlAlterDatabaseStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_AlterSchemaStatement(SqlAlterSchemaStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_CreatePrincipalStatement(SqlCreatePrincipalStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_StatisticsStatement(SqlStatisticsStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_AlterRoleStatement(SqlAlterRoleStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_AlterPrincipalStatement(SqlAlterPrincipalStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_DbccStatement(SqlDbccStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_BulkInsertStatement(SqlBulkInsertStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_AlterSequenceStatement(SqlAlterSequenceStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_KillStatement(SqlKillStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_BackupRestoreStatement(SqlBackupRestoreStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_CreateFulltextIndexStatement(SqlCreateFulltextIndexStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_CreatePartitionFunctionStatement(SqlCreatePartitionFunctionStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_CreatePartitionSchemeStatement(SqlCreatePartitionSchemeStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_CreateXmlSchemaCollectionStatement(SqlCreateXmlSchemaCollectionStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_OdbcEscapeExpr(SqlOdbcEscapeExpr expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_ToggleTriggerStatement(SqlToggleTriggerStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_AlterTableRebuild(SqlAlterTableRebuild expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_AlterTableSet(SqlAlterTableSet expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_AlterTableSwitch(SqlAlterTableSwitch expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_AlterAuthorizationStatement(SqlAlterAuthorizationStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_CreateCertificateStatement(SqlCreateCertificateStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_CreateMasterKeyStatement(SqlCreateMasterKeyStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_SetUserStatement(SqlSetUserStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_CreateFulltextCatalogStatement(SqlCreateFulltextCatalogStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_CreateFulltextStoplistStatement(SqlCreateFulltextStoplistStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_CreateSymmetricKeyStatement(SqlCreateSymmetricKeyStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_SymmetricKeyStatement(SqlSymmetricKeyStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_AlterFulltextIndexStatement(SqlAlterFulltextIndexStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_AlterServerConfigurationStatement(SqlAlterServerConfigurationStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_XmlNamespacesStatement(SqlXmlNamespacesStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_QuantifiedExpr(SqlQuantifiedExpr expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_CreateExternalStatement(SqlCreateExternalStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_CreateSecurityPolicyStatement(SqlCreateSecurityPolicyStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_CreateCredentialStatement(SqlCreateCredentialStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_AlterTablePeriod(SqlAlterTablePeriod expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_CreateRuleOrDefaultStatement(SqlCreateRuleOrDefaultStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_CreateAssemblyStatement(SqlCreateAssemblyStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_CreateAggregateStatement(SqlCreateAggregateStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_CreateQueueStatement(SqlCreateQueueStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_CreateServiceStatement(SqlCreateServiceStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_CreateContractStatement(SqlCreateContractStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_CreateMessageTypeStatement(SqlCreateMessageTypeStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_SendStatement(SqlSendStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_ReceiveStatement(SqlReceiveStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_CreateResourceGovernorObjectStatement(SqlCreateResourceGovernorObjectStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_CreateEndpointStatement(SqlCreateEndpointStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_CreateColumnKeyStatement(SqlCreateColumnKeyStatement expr)
    {
        AddSqlExpression(expr);
    }

    public virtual void Visit_TextPointerStatement(SqlTextPointerStatement expr)
    {
        AddSqlExpression(expr);
    }
}