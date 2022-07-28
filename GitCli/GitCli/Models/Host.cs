using Microsoft.Extensions.Hosting;

namespace GitCli.Models
{
   public class HostFactory
   {
      public IHostBuilder Create(string[] args)
      {
         return Host.CreateDefaultBuilder(args);
      }
   }
}
