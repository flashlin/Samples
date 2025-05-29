using System;
using System.Data;
using System.IO;
using ClosedXML.Excel;
using System.Collections.Generic;
using VimSharpApp.Databases;
using Microsoft.EntityFrameworkCore;

namespace VimSharpApp.Databases
{
    // Service for importing Excel to SQLite
    public class ExcelService
    {
        private readonly SupportDbRepo _repo;

        public ExcelService(SupportDbRepo repo)
        {
            _repo = repo;
        }

        // Import all sheets from excelFile to SQLite, each sheet as a table
        public void ImportExcel(string excelFile)
        {
            if (!File.Exists(excelFile))
                throw new FileNotFoundException($"Excel file not found: {excelFile}");

            using var workbook = new XLWorkbook(excelFile);
            var fileName = Path.GetFileNameWithoutExtension(excelFile);

            foreach (var worksheet in workbook.Worksheets)
            {
                var tableName = $"{fileName}_{worksheet.Name}";
                var dataTable = WorksheetToDataTable(worksheet);
                _repo.ImportSheetToDb(tableName, dataTable, excelFile);
            }
        }

        // Convert worksheet to DataTable
        private DataTable WorksheetToDataTable(IXLWorksheet worksheet)
        {
            var dt = new DataTable();
            bool firstRow = true;
            foreach (var row in worksheet.RowsUsed())
            {
                if (firstRow)
                {
                    foreach (var cell in row.Cells())
                        dt.Columns.Add(cell.GetString());
                    firstRow = false;
                }
                else
                {
                    var dataRow = dt.NewRow();
                    int i = 0;
                    foreach (var cell in row.Cells())
                        dataRow[i++] = cell.Value;
                    dt.Rows.Add(dataRow);
                }
            }
            return dt;
        }
    }
} 