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
        private readonly IApplicationWindow _applicationWindow;

        public Main(IApplicationWindow applicationWindow)
        {
            _applicationWindow = applicationWindow;
        }
        
        public Task Run()
        {
            _applicationWindow.Run();
            return Task.CompletedTask;
        }
    }
}