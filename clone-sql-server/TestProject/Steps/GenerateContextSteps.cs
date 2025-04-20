using CloneSqlServer;
using FluentAssertions;
using Microsoft.Data.SqlClient;
using Reqnroll;
using System.Threading.Tasks;

namespace TestProject.Steps
{
    [Binding]
    public class GenerateContextSteps
    {
        private GenerateContext? _context;
        private string _connectionString = "Server=127.0.0.1,1433;Database=master;User Id=sa;Password=YourStrongPassw0rd!;TrustServerCertificate=True;";

        [Given(@"I have connected to SQL Server")]
        public async Task GivenIHaveConnectedToSQLServer()
        {
            using var connection = new SqlConnection(_connectionString);
            await connection.OpenAsync();
            _context = await GenerateContext.Initialize(connection);
        }

        [Then(@"TableSchemas should contain correct table structure information")]
        public void ThenTableSchemasShouldContainCorrectTableStructureInformation()
        {
            _context.Should().NotBeNull();
            _context!.TableSchemas.Should().NotBeNull();
            _context.TableSchemas.Should().NotBeEmpty();
            
            foreach (var database in _context.TableSchemas.Keys)
            {
                var tableSchemas = _context.TableSchemas[database];
                tableSchemas.Should().NotBeNull();
                tableSchemas.Should().NotBeEmpty();
                
                foreach (var schema in tableSchemas)
                {
                    schema.TableName.Should().NotBeNullOrEmpty();
                    schema.ColumnName.Should().NotBeNullOrEmpty();
                    schema.DataType.Should().NotBeNullOrEmpty();
                }
            }
        }
    }
} 