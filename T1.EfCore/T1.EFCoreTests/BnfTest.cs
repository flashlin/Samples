using T1.EfCore.Parsers;

namespace T1.EFCoreTests;

public class BnfTest
{
    [Test]
    public void Test()
    {
        string bnfGrammar = @"
<expr> ::= <term> ""+"" <expr> | <term>
<term> ::= <factor> ""*"" <term> | <factor>
<factor> ::= ( <expr> ) | <number>
<number> ::= ""0"" | ""1"" | ""2"" | ""3"" | ""4"" | ""5"" | ""6"" | ""7"" | ""8"" | ""9""
";
        var parser = new BnfParser(bnfGrammar);
        var tree = parser.Parse();
        var text = parser.GetExpressionTreeString(tree);
    }
    
    [Test]
    public void LinqSyntax()
    {
        string bnfGrammar = @"
<linq_query> ::= ""from"" <range_variable> ""in"" <source> 
                 {<join_clause>}
                 {<where_clause>}
                 {<group_by_clause>}
                 {<order_by_clause>}
                 ""select"" <projection>

<range_variable> ::= <identifier>

<source> ::= <expression>

<join_clause> ::= ""join"" <range_variable> ""in"" <source> ""on"" <key_selector> ""equals"" <key_selector> 
                  {<join_clause>}

<where_clause> ::= ""where"" <condition>

<group_by_clause> ::= ""group"" <projection> ""by"" <key_selector>

<order_by_clause> ::= ""orderby"" <order_specifier> {"","" <order_specifier>}

<order_specifier> ::= <key_selector> [""ascending"" | ""descending""]

<projection> ::= <identifier> | ""new"" ""{"" <projection_list> ""}""

<projection_list> ::= <projection_item> {"","" <projection_item>}

<projection_item> ::= <expression> [""as"" <alias>]

<key_selector> ::= <expression>

<condition> ::= <expression>

<expression> ::= <term> {<operator> <term>}

<term> ::= <identifier> | <value> | <method_call> | <lambda_expression> | ""("" <expression> "")""

<operator> ::= ""=="" | ""!="" | ""<"" | "">"" | ""<="" | "">="" | ""&&"" | ""||""

<value> ::= <string_literal> | <numeric_literal>

<method_call> ::= <identifier> ""("" <argument_list> "")""

<lambda_expression> ::= ""("" <parameter_list> "")"" ""=>"" <expression>

<parameter_list> ::= <identifier> {"","" <identifier>}

<argument_list> ::= <expression> {"","" <expression>}

<identifier> ::= {<letter>} {<letter> | <digit> | ""_""}
";

        var parser = new BnfParser(bnfGrammar);
        var tree = parser.Parse();
        var text = parser.GetExpressionTreeString(tree);
        
        var visitor = new TsqlBnfExpressionVisitor();
        tree.Accept(visitor);
    }
}