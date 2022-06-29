using CommandLine;
using LanguageExt.Common;

namespace GitCli.Models;

public static class CommandParseExtension
{
	public static Result<T> ParseArgs<T>(this string[] args)
	{
		Result<T> opt = new Result<T>(new ParseArgumentsException());
		var parser = new Parser(configuration =>
		{
			configuration.AutoHelp = false;
			configuration.AutoVersion = false;
			configuration.CaseSensitive = false;
			configuration.IgnoreUnknownArguments = true;
			//configuration.HelpWriter = writer;
		});
		parser.ParseArguments(args, new Type[] { typeof(T) })
			.WithParsed(value =>
			{
				opt = new Result<T>((T)value);
			})
			.WithNotParsed(errs =>
			{
				//opt = new Result<T>(new ParseArgumentsException());
			});
		return opt;
		// var opts = Parser.Default.ParseArguments<T>(args)
		// 	 .MapResult(
		// 		 opts => new Result<T>(opts),
		// 		errs => new Result<T>(new ParseArgumentsException()));
		// return opts;
	}
}

public class ParseArgumentsException : Exception
{
	public ParseArgumentsException() : base()
	{
	}

	public ParseArgumentsException(string? message) : base(message)
	{
	}

	public ParseArgumentsException(string? message, Exception? innerException) : base(message, innerException)
	{
	}
}