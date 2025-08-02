using Microsoft.Data.SqlClient;

namespace CloneSqlServer.Kit;

public class SqlDbContext : IDisposable, IAsyncDisposable
{
    private SqlConnection? _connection;

    public async Task OpenAsync(string connectionString)
    {
        await CloseAsync();
        _connection = new SqlConnection(connectionString);
        await _connection.OpenAsync();
    }

    public async Task CloseAsync()
    {
        if (_connection == null)
        {
            return;
        }
        await _connection.CloseAsync();
    }

    public void Dispose()
    {
        _connection?.Dispose();
    }

    public async ValueTask DisposeAsync()
    {
        if (_connection != null)
        {
            await _connection.DisposeAsync();
        }
    }
}