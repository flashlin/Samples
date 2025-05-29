using System;
using System.Collections.Generic;
using System.Data;
using System.IO;
using System.Linq;
using Microsoft.EntityFrameworkCore;

namespace VimSharpApp.Databases
{
    // Repository for SupportDbContext
    public class SupportDbRepo
    {
        private SupportDbContext _context;

        public SupportDbRepo(SupportDbContext context)
        {
            _context = context;
        }

        // Execute query and return List<T>
        public List<T> QueryToList<T>(IQueryable<T> query)
        {
            // 將 IQueryable 轉成 List
            return query.ToList();
        }

        public void ImportSheetToDb(DataTable dt, string tableName)
        {
            DropTable(tableName);
            CreateTable(dt, tableName);
            InsertTable(dt, tableName);
        }

        private void InsertTable(DataTable dt, string tableName)
        {
            // 2. Insert data
            foreach (DataRow row in dt.Rows)
            {
                var colNames = string.Join(",", dt.Columns.Cast<DataColumn>().Select(c => $"[{c.ColumnName}]"));
                var values = string.Join(",", dt.Columns.Cast<DataColumn>().Select(c => $"@{c.ColumnName}"));
                var insertSql = $"INSERT INTO [{tableName}] ({colNames}) VALUES ({values})";
                var parameters = dt.Columns.Cast<DataColumn>()
                    .Select(c => new Microsoft.Data.Sqlite.SqliteParameter($"@{c.ColumnName}", row[c.ColumnName]?.ToString() ?? ""))
                    .ToArray();
                _context.Database.ExecuteSqlRaw(insertSql, parameters);
            }
        }

        private void DropTable(string tableName)
        {
            // Drop table if exists
            var dropTableSql = $"DROP TABLE IF EXISTS [{tableName}]";
            _context.Database.ExecuteSqlRaw(dropTableSql);
        }

        private void CreateTable(DataTable dt, string tableName)
        {
            var columns = new List<string>();
            foreach (DataColumn col in dt.Columns)
            {
                columns.Add($"[{col.ColumnName}] TEXT");
            }
            var createTableSql = $"CREATE TABLE [{tableName}] ({string.Join(",", columns)})";
            _context.Database.ExecuteSqlRaw(createTableSql);
        }
    }
} 