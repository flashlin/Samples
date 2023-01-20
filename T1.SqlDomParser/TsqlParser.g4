parser grammar TsqlParser;

options {
	// 表示解析token的词法解析器使用SearchLexer
	tokenVocab = TsqlLexer;
}

select_statement:
	SELECT allOrDistinct = (ALL | DISTINCT)? top = top_clause? columns = select_list;

top_clause: TOP (top_count);

top_count:
	count_constant = DECIMAL; //| '(' topcount_expression=expression ')'

column_elem: (full_column_name) as_column_alias?;

full_column_name: (DELETED | INSERTED) '.' column_name = id_
	| server = id_? '.' schema = id_? '.' tablename = id_? '.' column_name = id_
	| schema = id_? '.' tablename = id_? '.' column_name = id_
	| tablename = id_? '.' column_name = id_
	| column_name = id_;

select_list: selectElement += column_elem (COMMA column_elem)*;

id_: ID_LETTER | SQUARE_BRACKET_ID;

as_column_alias: AS? column_alias;

column_alias: id_ | STRING;
