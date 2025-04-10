﻿using FluentAssertions;
using ParserKitTests.Helpers;
using T1.ParserKit;
using T1.ParserKit.ExprCollection;

namespace ParserKitTests
{
    public class SqlParserTest
    {
        private SqlParser _sut = null!;

        [SetUp]
        public void SetUp()
        {
            _sut = new SqlParser();
        }

        [Test]
        public void Select_field()
        {
            var expr = _sut.Parse("select id from customer");

            expr.ShouldAllSatisfy(
                new SelectExpr
                {
                    Columns = new List<SqlExpr>()
                    {
                        new FieldExpr
                        {
                            Name = "id"
                        }
                    },
                    FromClause = new TableExpr
                    {
                        Name = "customer"
                    }
                }
            );
        }

        [Test]
        public void Select_field_aliasName()
        {
            var expr = _sut.Parse("select id id1 from customer") as SelectExpr;

            expr.ShouldAllSatisfy(new SelectExpr
            {
                Columns = new List<SqlExpr>()
                {
                    new FieldExpr
                    {
                        Name = "id",
                        AliasName = "id1"
                    },
                },
                FromClause = new TableExpr
                {
                    Name = "customer",
                    AliasName = string.Empty
                }
            });
        }

        [Test]
        public void Select_field_as_aliasName()
        {
            var expr = _sut.Parse("select id as id1 from customer") as SelectExpr;

            expr.ShouldAllSatisfy(new SelectExpr
            {
                Columns = new List<SqlExpr>()
                {
                    new FieldExpr
                    {
                        Name = "id",
                        AliasName = "id1"
                    },
                },
                FromClause = new TableExpr
                {
                    Name = "customer",
                    AliasName = string.Empty
                }
            });
        }

        [Test]
        public void Select_field_as_aliasName_from_table_asTableName()
        {
            var expr = _sut.Parse("select id as id1 from customer c") as SelectExpr;

            expr.ShouldAllSatisfy(new SelectExpr
            {
                Columns = new List<SqlExpr>()
                {
                    new FieldExpr
                    {
                        Name = "id",
                        AliasName = "id1"
                    },
                },
                FromClause = new TableExpr
                {
                    Name = "customer",
                    AliasName = "c"
                }
            });
        }

        [Test]
        public void Select_field_as_aliasName_from_select_field_from_table()
        {
            var expr = _sut.Parse("select id as id1 from (select cid id from extraCustomer) c1") as SelectExpr;

            expr.ShouldAllSatisfy(new SelectExpr
            {
                Columns = new List<SqlExpr>()
                {
                    new FieldExpr
                    {
                        Name = "id",
                        AliasName = "id1"
                    },
                },
                FromClause = new FromSourceExpr
                {
                    Clause = new SelectExpr
                    {
                        Columns = new List<SqlExpr>()
                        {
                            new FieldExpr
                            {
                                Name = "cid",
                                AliasName = "id"
                            }
                        },
                        FromClause = new TableExpr
                        {
                            Name = "extraCustomer",
                            AliasName = ""
                        }
                    },
                    AliasName = "c1"
                }
            });
        }
    }
}