﻿using FluentAssertions;
using T1.ParserKit;
using T1.Standard.Collections.Generics;

namespace ParserKitTests;

public class TokenizeTest
{
    [Test]
    public void Where()
    {
        var sql = "WHERE ID=1 and (Name !='' or Name like '%name%')";
        var tk = new Tokenizer();
        var tokens = tk.Tokenize(sql);
        tokens.Should().BeEquivalentTo(new[]
        {
            "WHERE", " ",
            "ID", "=", "1", " ",
            "and", " ",
            "(", "Name", " ", "!=", "'", "'", " ",
            "or", " ", "Name", " ", "like", " ",
            "'", "%", "name", "%", "'", ")"
        }.Select(x => new Token
        {
            Text = x,
        }), options => options.Including(o => o.Text));
    }

    [Test]
    public void SqlTokenize()
    {
        var sql = "WHERE ID=1 and (Name !='' or Name like '%name%')";
        var tk = new SqlTokenizer();
        var tokens = tk.Tokenize(sql);
        tokens.Should().BeEquivalentTo(new[]
        {
            "WHERE", "ID", "=", "1",
            "and",
            "(", "Name", "!=", "''",
            "or", "Name", "like",
            "'%name%'", ")"
        }.Select(x => new Token
        {
            Text = x,
        }), options => options.Including(o => o.Text));
    }
}