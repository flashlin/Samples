using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.EntityFrameworkCore;

namespace VimSharpApp.Databases
{
    // Repository for SupportDbContext
    public class SupportDbRepo
    {
        // Create SQLite .db file if not exists
        public void CreateDbFile(string dbFile)
        {
            if (!File.Exists(dbFile))
            {
                // 建立空的 SQLite 檔案
                using var context = new SupportDbContext(dbFile);
                context.Database.EnsureCreated();
            }
        }

        // Execute query and return List<T>
        public List<T> QueryToList<T>(IQueryable<T> query)
        {
            // 將 IQueryable 轉成 List
            return query.ToList();
        }
    }
} 