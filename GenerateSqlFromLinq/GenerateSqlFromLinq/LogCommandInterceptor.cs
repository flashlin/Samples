using System.Collections.Concurrent;
using System.Data.Common;
using Microsoft.EntityFrameworkCore.Diagnostics;

class LogCommandInterceptor : DbCommandInterceptor
{
	private ConcurrentQueue<string> _log;

	public LogCommandInterceptor(ConcurrentQueue<string> log)
	{
		_log = log;
	}

	void WriteLine(CommandEventData data)
	{
		_log.Enqueue(data.Command.CommandText);
		//Trace.WriteLine($@"EF {data}", "LocalDB");
	}

	public override void CommandFailed(DbCommand command, CommandErrorEventData data)
	{
		WriteLine(data);
	}

	public override Task CommandFailedAsync(DbCommand command, CommandErrorEventData data, CancellationToken cancellation)
	{
		WriteLine(data);
		return Task.CompletedTask;
	}

	public override DbDataReader ReaderExecuted(DbCommand command, CommandExecutedEventData data, DbDataReader result)
	{
		WriteLine(data);
		return result;
	}

	public override object ScalarExecuted(DbCommand command, CommandExecutedEventData data, object result)
	{
		WriteLine(data);
		return result;
	}

	public override int NonQueryExecuted(DbCommand command, CommandExecutedEventData data, int result)
	{
		WriteLine(data);
		return result;
	}

	public override ValueTask<DbDataReader> ReaderExecutedAsync(DbCommand command, CommandExecutedEventData data, DbDataReader result, CancellationToken cancellation)
	{
		WriteLine(data);
		return ValueTask.FromResult(result);
	}

	public override ValueTask<object> ScalarExecutedAsync(DbCommand command, CommandExecutedEventData data, object result, CancellationToken cancellation)
	{
		WriteLine(data);
		return ValueTask.FromResult(result);
	}

	public override ValueTask<int> NonQueryExecutedAsync(DbCommand command, CommandExecutedEventData data, int result, CancellationToken cancellation)
	{
		WriteLine(data);
		return ValueTask.FromResult(result);
	}
}