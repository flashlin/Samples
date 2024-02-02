using System.Text;
using System.Xml;
using System.Xml.Linq;

namespace ResxParserDemo;

public class ResxParser
{
    public void ReadFile(string resxFile)
    {
        var t = File.ReadAllText(resxFile, Encoding.UTF8);
        using var sr = new StreamReader(resxFile, Encoding.UTF8);
        var xDoc = XDocument.Load(sr);
        foreach (var dataNode in xDoc.Descendants("data"))
        {
            var name = dataNode.Attribute("name")!.Value;
            var value = dataNode.Element("value")!.Value;
            Console.WriteLine($"{name} = {value}");
        }
    }
}