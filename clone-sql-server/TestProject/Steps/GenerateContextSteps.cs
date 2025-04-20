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
            _context.TableSchemas.Should().NotBeEmpty();
            
            foreach (var database in _context.TableSchemas.Keys)
            {
                var tableSchemas = _context.TableSchemas[database];
                tableSchemas.Should().NotBeNull();
                tableSchemas.Should().NotBeEmpty();
                
                // 驗證 Categories 表格
                var categoriesSchema = tableSchemas.Where(s => s.TableName == "Categories").ToList();
                categoriesSchema.Should().NotBeEmpty();
                categoriesSchema.Should().Contain(s => s.ColumnName == "CategoryId" && s.DataType == "int" && s.IsIdentity);
                categoriesSchema.Should().Contain(s => s.ColumnName == "CategoryName" && s.DataType == "nvarchar" && s.CharacterMaxLength == 50 && !s.IsNullable);
                categoriesSchema.Should().Contain(s => s.ColumnName == "CreatedDate" && s.DataType == "datetime" && s.IsNullable);

                // 驗證 Products 表格
                var productsSchema = tableSchemas.Where(s => s.TableName == "Products").ToList();
                productsSchema.Should().NotBeEmpty();
                productsSchema.Should().Contain(s => s.ColumnName == "ProductId" && s.DataType == "int" && s.IsIdentity);
                productsSchema.Should().Contain(s => s.ColumnName == "CategoryId" && s.DataType == "int" && !s.IsNullable);
                productsSchema.Should().Contain(s => s.ColumnName == "ProductName" && s.DataType == "nvarchar" && s.CharacterMaxLength == 100 && !s.IsNullable);
                productsSchema.Should().Contain(s => s.ColumnName == "UnitPrice" && s.DataType == "decimal" && s.NumericPrecision == 18 && s.NumericScale == 2 && s.IsNullable);
                productsSchema.Should().Contain(s => s.ColumnName == "CreatedDate" && s.DataType == "datetime" && s.IsNullable);

                // 驗證 OrderDetails 表格
                var orderDetailsSchema = tableSchemas.Where(s => s.TableName == "OrderDetails").ToList();
                orderDetailsSchema.Should().NotBeEmpty();
                orderDetailsSchema.Should().Contain(s => s.ColumnName == "OrderDetailId" && s.DataType == "int" && s.IsIdentity);
                orderDetailsSchema.Should().Contain(s => s.ColumnName == "ProductId" && s.DataType == "int" && !s.IsNullable);
                orderDetailsSchema.Should().Contain(s => s.ColumnName == "Quantity" && s.DataType == "int" && !s.IsNullable);
                orderDetailsSchema.Should().Contain(s => s.ColumnName == "OrderDate" && s.DataType == "datetime" && s.IsNullable);

                // 驗證 LogEvents 表格
                var logEventsSchema = tableSchemas.Where(s => s.TableName == "LogEvents").ToList();
                logEventsSchema.Should().NotBeEmpty();
                logEventsSchema.Should().Contain(s => s.ColumnName == "LogId" && s.DataType == "int" && s.IsIdentity);
                logEventsSchema.Should().Contain(s => s.ColumnName == "EventType" && s.DataType == "nvarchar" && s.CharacterMaxLength == 50 && !s.IsNullable);
                logEventsSchema.Should().Contain(s => s.ColumnName == "EventMessage" && s.DataType == "nvarchar" && s.CharacterMaxLength == -1 && s.IsNullable);
                logEventsSchema.Should().Contain(s => s.ColumnName == "CreatedDate" && s.DataType == "datetime" && s.IsNullable);
            }
        }
    }
} 