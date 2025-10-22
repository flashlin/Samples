using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using T1.EfCodeFirstGenerateCli.Models;

namespace T1.EfCodeFirstGenerateCli.ConfigParser
{
    public class DbConfigParser
    {
        public static List<DbConfig> GetAllDbConnectionConfigs(string directory)
        {
            var configs = new List<DbConfig>();
            var dbFiles = Directory.GetFiles(directory, "*.db", SearchOption.AllDirectories);

            foreach (var dbFile in dbFiles)
            {
                var lines = File.ReadAllLines(dbFile);
                var fileName = Path.GetFileNameWithoutExtension(dbFile);
                
                foreach (var line in lines)
                {
                    var lineText = line.Trim();
                    if (string.IsNullOrWhiteSpace(lineText) || lineText.StartsWith("#") || lineText.StartsWith("//"))
                        continue;

                    var config = ParseConnectionString(lineText);
                    if (config != null)
                    {
                        config.DbFilePath = dbFile;
                        config.ContextName = fileName;
                        configs.Add(config);
                        break;
                    }
                }
            }

            return configs;
        }

        public static void ProcessAllConfigs(
            string directory,
            Action<DbConfig> processAction,
            Action<string>? logAction = null)
        {
            var dbConfigs = GetAllDbConnectionConfigs(directory);

            if (dbConfigs.Count == 0)
            {
                logAction?.Invoke("No .db files found or no valid connection strings.");
                return;
            }

            logAction?.Invoke($"Found {dbConfigs.Count} database configuration(s).");

            foreach (var dbConfig in dbConfigs)
            {
                processAction(dbConfig);
            }
        }

        private static DbConfig? ParseConnectionString(string connectionString)
        {
            var config = new DbConfig();
            var parts = SplitConnectionString(connectionString);

            foreach (var part in parts)
            {
                var keyValue = part.Split(new[] { '=' }, 2);
                if (keyValue.Length != 2) continue;

                var key = keyValue[0].Trim().ToLower();
                var value = keyValue[1].Trim();

                switch (key)
                {
                    case "server":
                    case "data source":
                    case "datasource":
                    case "host":
                        config.ServerName = value;
                        break;
                    case "database":
                    case "initial catalog":
                    case "initialcatalog":
                        config.DatabaseName = value;
                        break;
                    case "user id":
                    case "userid":
                    case "uid":
                    case "username":
                    case "user":
                        config.LoginName = value;
                        break;
                    case "password":
                    case "pwd":
                        config.Password = value;
                        break;
                    default:
                        if (!string.IsNullOrEmpty(config.ExtraInfo))
                            config.ExtraInfo += ";";
                        config.ExtraInfo += part;
                        break;
                }
            }

            if (string.IsNullOrEmpty(config.ServerName) || string.IsNullOrEmpty(config.DatabaseName))
            {
                return null;
            }

            config.DbType = DetectDbType(connectionString);
            return config;
        }

        private static List<string> SplitConnectionString(string connectionString)
        {
            var parts = new List<string>();
            var current = "";
            var inQuotes = false;

            for (int i = 0; i < connectionString.Length; i++)
            {
                var ch = connectionString[i];
                
                if (ch == '\'' || ch == '"')
                {
                    inQuotes = !inQuotes;
                    current += ch;
                }
                else if (ch == ';' && !inQuotes)
                {
                    if (!string.IsNullOrWhiteSpace(current))
                        parts.Add(current.Trim());
                    current = "";
                }
                else
                {
                    current += ch;
                }
            }

            if (!string.IsNullOrWhiteSpace(current))
                parts.Add(current.Trim());

            return parts;
        }

        private static DbType DetectDbType(string connectionString)
        {
            var lower = connectionString.ToLower();
            
            if (lower.Contains("mysql") || lower.Contains("mariadb"))
                return DbType.MySql;
            else if (lower.Contains("postgres") || lower.Contains("npgsql"))
                return DbType.PostgreSql;
            else if (lower.Contains("oracle"))
                return DbType.Oracle;
            else
                return DbType.SqlServer;
        }
    }
}

