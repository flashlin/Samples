using System.Collections.Generic;
using System.Linq;
using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
	public class CreateParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			if (parser.TryConsumeToken(out var nonclusteredSpan, SqlToken.NONCLUSTERED))
			{
				return CreateNonclusteredIndex(nonclusteredSpan, parser);
			}
			
			if (parser.TryConsumeToken(out var synonymSpan, SqlToken.SYNONYM))
			{
				return CreateSynonym(synonymSpan, parser);
			}
			
			if(parser.Scanner.TryConsume(SqlToken.CLUSTERED, out var clusteredSpan))
			{
				return CreateClusteredIndex(clusteredSpan, parser);
			}
			
			if(parser.Scanner.TryConsume(SqlToken.Index, out var indexSpan))
			{
				return CreateIndex(indexSpan, parser);
			}

			if (parser.Scanner.TryConsume(SqlToken.TABLE, out var tableSpan))
			{
				return CreateTable(tableSpan, parser);
			}

			if (parser.Scanner.Match(SqlToken.Procedure))
			{
				return CreateProcedure(token, parser);
			}

			if (parser.Scanner.Match(SqlToken.Function))
			{
				return CreateFunction(token, parser);
			}

			if (parser.Scanner.Match(SqlToken.Partition))
			{
				return CreatePartitionFunction(token, parser);
			}

			var helpMessage = parser.Scanner.GetHelpMessage();
			throw new ParseException($"Parse CREATE Error, {helpMessage}");
		}

		private SqlCodeExpr CreateNonclusteredIndex(TextSpan nonclusteredSpan, IParser parser)
		{
			parser.ConsumeToken(SqlToken.Index);
			var indexName = parser.ConsumeObjectId();
			parser.ConsumeToken(SqlToken.ON);
			var tableName = parser.ConsumeObjectId();
			parser.ConsumeToken(SqlToken.LParen);
			var columnList = parser.ParseOrderItemList();
			parser.ConsumeToken(SqlToken.RParen);

			var withExpr = parser.ParseConstraintWithOptions();
			var onPrimary = parser.ParseOnPrimary();
			
			var isSemicolon = parser.MatchToken(SqlToken.Semicolon);
			return new CreateNonclusteredIndexSqlCodeExpr
			{
				IndexName = indexName,
				TableName = tableName,
				ColumnList = columnList,
				WithExpr = withExpr,
				OnPrimary = onPrimary,
				IsSemicolon = isSemicolon,
			};
		}

		private SqlCodeExpr CreateSynonym(TextSpan synonymSpan, IParser parser)
		{
			var synonymName = parser.ConsumeObjectId();
			parser.ConsumeToken(SqlToken.FOR);
			var objectId = parser.ConsumeObjectId();
			return new CreateSynonymSqlCodeExpr
			{
				Name = synonymName,
				ObjectId = objectId
			};
		}

		private SqlCodeExpr CreateClusteredIndex(TextSpan clusteredSpan, IParser parser)
		{
			parser.Scanner.Consume(SqlToken.Index);
			var indexName = parser.ConsumeObjectId();
			parser.Scanner.Consume(SqlToken.ON);
			
			if( !parser.TryConsumeObjectId(out var tableName) )
			{
				tableName = parser.Consume(SqlToken.TempTable);
			}
			
			parser.Scanner.Consume(SqlToken.LParen);
			var onColumnsList = parser.ParseOrderItemList();
			parser.Scanner.Consume(SqlToken.RParen);
			
			var withExpr = parser.ParseConstraintWithOptions();

			return new CreateClusteredIndexSqlCodeExpr
			{
				IndexName = indexName,
				TableName = tableName,
				OnColumns = onColumnsList,
				WithExpr = withExpr
			};
		}
		
		private SqlCodeExpr CreateIndex(TextSpan indexSpan, IParser parser)
		{
			var indexName = parser.ConsumeObjectId();
			parser.Scanner.Consume(SqlToken.ON);
			
			if( !parser.TryConsumeObjectId(out var tableName) )
			{
				tableName = parser.Consume(SqlToken.TempTable);
			}
			
			var onColumnsList = new List<SqlCodeExpr>();
			parser.Scanner.Consume(SqlToken.LParen);
			do
			{
				var columnName = parser.ConsumeObjectId();
				onColumnsList.Add(columnName);
			} while (parser.Scanner.Match(SqlToken.Comma));
			parser.Scanner.Consume(SqlToken.RParen);

			return new CreateIndexSqlCodeExpr
			{
				IndexName = indexName,
				TableName = tableName,
				OnColumns = onColumnsList
			};
		}

		private SqlCodeExpr CreateTable(TextSpan tableSpan, IParser parser)
		{
			var tableName = parser.ConsumeTableName();

			var tableType = parser.ConsumeTableDataType();
			tableType.Name = tableName;
			
			var onPrimary = parser.ParseOnPrimary();

			var isSemicolon = parser.MatchToken(SqlToken.Semicolon);

			return new CreateTableSqlCodeExpr
			{
				TableExpr = tableType,
				OnPrimary = onPrimary,
				IsSemicolon = isSemicolon
			};
		}

		private IExpression CreatePartitionFunction(TextSpan token, IParser parser)
		{
			if (parser.Scanner.IsToken(SqlToken.Scheme))
			{
				return CreatePartitionScheme(token, parser);
			}

			parser.Scanner.Consume(SqlToken.Function);

			var name = parser.ConsumeObjectId();

			parser.Scanner.Consume(SqlToken.LParen);
			var dataType = parser.ConsumeDataType();
			parser.Scanner.Consume(SqlToken.RParen);

			parser.Scanner.Consume(SqlToken.As);
			parser.Scanner.Consume(SqlToken.Range);

			var rangeType = string.Empty;
			if (parser.Scanner.TryConsumeAny(out var rangeTypeSpan, SqlToken.Left, SqlToken.Right))
			{
				rangeType = parser.Scanner.GetSpanString(rangeTypeSpan);
			}

			parser.Scanner.Consume(SqlToken.FOR);
			parser.Scanner.Consume(SqlToken.Values);

			var boundaryValueList = new List<SqlCodeExpr>();
			parser.Scanner.Consume(SqlToken.LParen);
			do
			{
				var boundaryValue = parser.ParseExpIgnoreComment();
				boundaryValueList.Add(boundaryValue);
			} while (parser.Scanner.Match(SqlToken.Comma));
			parser.Scanner.Consume(SqlToken.RParen);

			return new CreatePartitionFunctionSqlCodeExpr
			{
				Name = name,
				DataType = dataType,
				RangeType = rangeType,
				BoundaryValueList = boundaryValueList,
			};
		}

		private IExpression CreatePartitionScheme(TextSpan token, IParser parser)
		{
			parser.Scanner.Consume(SqlToken.Scheme);
			var schemeName = parser.ConsumeObjectId();

			parser.Scanner.Consume(SqlToken.As);
			parser.Scanner.Consume(SqlToken.Partition);

			var funcName = parser.ConsumeObjectId();

			parser.Scanner.TryConsumeString(SqlToken.All, out var allToken);
			parser.Scanner.Consume(SqlToken.To);

			var groupNameList = new List<SqlCodeExpr>();
			parser.Scanner.Consume(SqlToken.LParen);
			do
			{
				groupNameList.Add(parser.ConsumePrimary());
			} while (parser.Scanner.Match(SqlToken.Comma));
			parser.Scanner.Consume(SqlToken.RParen);

			return new CreatePartitionSchemeSqlCodeExpr
			{
				SchemeName = schemeName,
				FuncName = funcName,
				AllToken = allToken,
				GroupNameList = groupNameList
			};
		}

		private IExpression CreateFunction(TextSpan token, IParser parser)
		{
			var nameExpr = parser.ConsumeObjectId();
			parser.Scanner.Consume(SqlToken.LParen);
			var arguments = parser.ConsumeArgumentList();
			parser.Scanner.Consume(SqlToken.RParen);

			parser.Scanner.Consume(SqlToken.Returns);

			parser.Scanner.TryConsumeVariable(out var returnVariableExpr);


			var returnTypeExpr = parser.ConsumeDataType();

			parser.Scanner.Consume(SqlToken.As);

			var body = parser.ConsumeBeginBody();

			return new CreateFunctionSqlCodeExpr
			{
				Name = nameExpr,
				Arguments = arguments,
				ReturnVariable = returnVariableExpr,
				ReturnType = returnTypeExpr,
				Body = body
			};
		}

		private IExpression CreateProcedure(TextSpan token, IParser parser)
		{
			var nameExpr = parser.ConsumeObjectId();
			var arguments = parser.ConsumeArgumentList();

			SqlCodeExpr withExecuteAsExpr = null;
			if (parser.MatchTokenList(SqlToken.With, SqlToken.Execute, SqlToken.As))
			{
				var userExpr =
					parser.ConsumeTokenStringAny(SqlToken.CALLER, SqlToken.SELF, SqlToken.OWNER, SqlToken.QuoteString);
				withExecuteAsExpr = new WithExecuteAsSqlCodeExpr
				{
					UserExpr = userExpr
				};
			}
			
			parser.Scanner.Consume(SqlToken.As);
			var bodyList = parser.ConsumeBeginBodyOrSingle();

			return new CreateProcedureSqlCodeExpr
			{
				Name = nameExpr,
				Arguments = arguments,
				WithExecuteAs = withExecuteAsExpr,
				Body = bodyList
			};
		}
	}

	public class CreateNonclusteredIndexSqlCodeExpr : SqlCodeExpr
	{
		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("CREATE NONCLUSTERED INDEX ");
			IndexName.WriteToStream(stream);
			stream.Write(" ON ");
			TableName.WriteToStream(stream);
			stream.Write("(");
			ColumnList.WriteToStreamWithComma(stream);
			stream.Write(")");

			if (WithExpr != null)
			{
				stream.Write(" ");
				WithExpr.WriteToStream(stream);
			}

			if (OnPrimary != null)
			{
				stream.Write(" ");
				OnPrimary.WriteToStream(stream);
			}

			if (IsSemicolon)
			{
				stream.Write(" ;");
			}
		}

		public SqlCodeExpr IndexName { get; set; }
		public SqlCodeExpr TableName { get; set; }
		public List<OrderItemSqlCodeExpr> ColumnList { get; set; }
		public bool IsSemicolon { get; set; }
		public SqlCodeExpr WithExpr { get; set; }
		public OnSqlCodeExpr OnPrimary { get; set; }
	}

	public class CreateSynonymSqlCodeExpr : SqlCodeExpr
	{
		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("CREATE SYNONYM ");
			Name.WriteToStream(stream);
			stream.Write(" FOR ");
			ObjectId.WriteToStream(stream);
		}

		public SqlCodeExpr Name { get; set; }
		public SqlCodeExpr ObjectId { get; set; }
	}

	public class WithExecuteAsSqlCodeExpr : SqlCodeExpr
	{
		public override void WriteToStream(IndentStream stream)
		{
			stream.Write($"WITH EXECUTE AS {UserExpr}");
		}

		public string UserExpr { get; set; }
	}
}