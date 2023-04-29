using FluentAssertions;
using QueryKits.CsvEx;
using QueryKits.Services;
using T1.SqlLocalData;
using T1.Standard.DynamicCode;

namespace QueryKitsTests;

public class TextConverterTest
{
    private readonly TextConverter _converter = new();

    [Test]
    public void Tab()
    {
        var text = @"a	b
1	n1";

        var csvSheet = _converter.ConvertTextToCsvSheet(text);

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

        var csvSheet = _converter.ConvertTextToCsvSheet(text);

        csvSheet.Headers.Should()
            .BeEquivalentTo(new List<CsvHeader>()
            {
                new() {ColumnType = ColumnType.Number, Name = "a"},
                new() {ColumnType = ColumnType.String, Name = "b"}
            });
    }
    
    
    [Test]
    public void SaveToCsvString()
    {
        var text = @"a,b
1,n1
";

        var csvSheet = _converter.ConvertTextToCsvSheet(text);
        
        var csv = csvSheet.SaveToString();

        csv.Should().Be(text);
    }
    
    
    [Test]
    public void SaveTabToCsvString()
    {
        var text = @"a	b
1	n1
";

        var csvSheet = _converter.ConvertTextToCsvSheet(text);
        csvSheet.Delimiter = ",";
        var csv = csvSheet.SaveToString();

        csv.Should().Be("a,b\r\n1,n1\r\n");
    }
    
    
    [Test]
    public void SaveSpaceToCsvString()
    {
        var text = @"a b
1 n1
2 n2";

        var csvSheet = _converter.ConvertTextToCsvSheet(text);
        csvSheet.Delimiter = ",";
        var csv = csvSheet.SaveToString();

        csv.Should().Be("a,b\r\n1,n1\r\n2,n2\r\n");
    }
    
}