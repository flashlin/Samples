using System.Net.NetworkInformation;

namespace QueryWeb.Models;

public class NetworkHelper
{
    public void DisplayListenPorts()
    {
        Console.WriteLine("Checking TCP");
        var properties = IPGlobalProperties.GetIPGlobalProperties();
        var connections = properties.GetActiveTcpConnections();
        foreach (var connection in connections)
        {
            if (connection.State == TcpState.Listen)
            {
                Console.WriteLine("Listen Port: " + connection.LocalEndPoint.Port);
            }
            else
            {
                Console.WriteLine("No Listen Port: " + connection.LocalEndPoint.Port);
            }
        }
    }
}