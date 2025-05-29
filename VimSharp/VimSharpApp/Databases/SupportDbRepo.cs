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

        public void ImportSheetToDb(string tableName, DataTable dt, string dbFile)
        {
            // 1. Create table if not exists
            var columns = new List<string>();
            foreach (DataColumn col in dt.Columns)
            {
                columns.Add($"[{col.ColumnName}] TEXT");
            }
                
            var createTableSql = $"CREATE TABLE IF NOT EXISTS [{tableName}] ({string.Join(",", columns)})";
            _context.Database.ExecuteSqlRaw(createTableSql);

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
    }
} 