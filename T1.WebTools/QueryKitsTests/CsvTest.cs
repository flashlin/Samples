using FluentAssertions;
using QueryKits.CsvEx;
using T1.SqlLocalData;
using T1.Standard.DynamicCode;

namespace QueryKitsTests;

public class CsvTest
{
    [Test]
    public void Tab()
    {
        var text = @"a	b
1	n1";

        var csvSheet = CsvSheet.ReadFromString(text);

        csvSheet.Headers.Should()
            .BeEquivalentTo(new List<CsvHeader>()
            {
                new() {ColumnType = ColumnType.Number, Name = "a"},
                new() {ColumnType = ColumnType.String, Name = "b"}
            });
    }
    
    
    [Test]
    public void Comma()
    {
        var text = @"a,b
1,n1";

        var csvSheet = CsvSheet.ReadFromString(text);

        csvSheet.Headers.Should()
            .BeEquivalentTo(new List<CsvHeader>()
            {
                new() {ColumnType = ColumnType.Number, Name = "a"},
                new() {ColumnType = ColumnType.String, Name = "b"}
            });
    }
    
    
}