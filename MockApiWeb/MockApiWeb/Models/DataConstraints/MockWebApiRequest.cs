using System.Runtime.Serialization;

namespace MockApiWeb.Models.DataConstraints;

[DataContract]
public class MockWebApiRequest
{
    [DataMember(Name="ProductName")]
    public string ProductName { get; set; } = string.Empty;
    
    [DataMember(Name="RequestBody")]
    public string RequestBody { get; set; } = string.Empty;
    
    [DataMember(Name="ControllerName")]
    public string ControllerName { get; set; } = string.Empty;
    
    [DataMember(Name="ActionName")]
    public string ActionName { get; set; } = string.Empty;
    
    [DataMember(Name="RequestQueryString")]
    public string RequestQueryString { get; set; } = string.Empty;
}