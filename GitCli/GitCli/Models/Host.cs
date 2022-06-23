using Microsoft.Extensions.Hosting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
