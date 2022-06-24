using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NStack;
using GitCli.Models;

namespace GitCli.Models
{
    public class Main
    {
        public Task Run()
        {
            new ApplicationWindow().Run();
            return Task.CompletedTask;
        }
    }
}